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
import pathlib
from pathlib import Path
from unittest import mock

from tools import install_gpu_profile as installer


class ReviewLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._conda_python_source = (
            b"fixture-python-prefix-before\0"
            + installer.CONDA_PYTHON_PREFIX_PLACEHOLDER
            + b"\0fixture-python-prefix-after\n"
        )
        self._python_identity = mock.patch.multiple(
            installer,
            CONDA_PYTHON_BUILD="fixture_0_cpython",
            CONDA_PYTHON_ARCHIVE_SHA256="a" * 64,
            CONDA_PYTHON_ARCHIVE_SIZE_BYTES=len(self._conda_python_source) + 1024,
            CONDA_PYTHON_BINARY_SOURCE_SHA256=hashlib.sha256(
                self._conda_python_source
            ).hexdigest(),
            CONDA_PYTHON_BINARY_SIZE_BYTES=len(self._conda_python_source),
        )
        self._python_identity.start()
        self.addCleanup(self._python_identity.stop)

    def _fixture(self, root: Path, profile: str) -> tuple[Path, Path]:
        venv = root / "exact environment"
        binary = venv / "bin/python"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)

        project = root / "initialized project"
        (project / "review").mkdir(parents=True)
        (project / "config.toml").write_text(
            "schema = \"compag-curation-public-project/v1\"\n"
            "[execution]\n"
            f"profile = \"{profile}\"\n",
            encoding="utf-8",
        )
        project.chmod(0o755)
        (project / "review").chmod(0o755)
        (project / "config.toml").chmod(0o644)
        return venv, project

    def _group_writable_conda_fixture(self, root: Path, profile: str) -> tuple[Path, Path]:
        """Model the fresh Miniforge/mamba Python layout observed on Ubuntu."""

        venv = root / "exact conda environment"
        binary = venv / "bin/python3.12"
        binary.parent.mkdir(parents=True)
        encoded_prefix = os.fsencode(venv)
        placeholder = installer.CONDA_PYTHON_PREFIX_PLACEHOLDER
        self.assertLessEqual(len(encoded_prefix), len(placeholder))
        relocated = encoded_prefix + b"\0" * (len(placeholder) - len(encoded_prefix))
        binary.write_bytes(self._conda_python_source.replace(placeholder, relocated, 1))
        binary.chmod(0o775)
        (venv / "bin/python").symlink_to("python3.12")

        preset = next(
            name for name, candidate in installer.PRESET_PROFILES.items() if candidate == profile
        )
        receipt = venv / installer.RECEIPT_RELATIVE
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "schema": installer.INSTALL_RECEIPT_SCHEMA,
                    "status": "PASS",
                    "release_version": installer.RELEASE_VERSION,
                    "preset": preset,
                    "profile": profile,
                    "dependency_profile": "science-gpu",
                    "venv": str(venv),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        receipt.chmod(0o644)

        metadata_directory = venv / "conda-meta"
        metadata_directory.mkdir()
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        build = installer.CONDA_PYTHON_BUILD
        metadata = metadata_directory / f"python-3.12.7-{build}.json"
        metadata.write_text(
            json.dumps(
                {
                    "name": "python",
                    "version": "3.12.7",
                    "build": build,
                    "build_number": 0,
                    "fn": f"python-3.12.7-{build}.conda",
                    "sha256": installer.CONDA_PYTHON_ARCHIVE_SHA256,
                    "size": installer.CONDA_PYTHON_ARCHIVE_SIZE_BYTES,
                    "subdir": "linux-64",
                    "paths_data": {
                        "paths_version": 1,
                        "paths": [
                            {
                                "_path": "bin/python3.12",
                                "path_type": "hardlink",
                                "sha256": installer.CONDA_PYTHON_BINARY_SOURCE_SHA256,
                                "sha256_in_prefix": digest,
                                "size_in_bytes": len(binary.read_bytes()),
                            }
                        ],
                    },
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata.chmod(0o644)

        project = root / "initialized conda project"
        (project / "review").mkdir(parents=True)
        (project / "config.toml").write_text(
            "schema = \"compag-curation-public-project/v1\"\n"
            "[execution]\n"
            f"profile = \"{profile}\"\n",
            encoding="utf-8",
        )
        project.chmod(0o755)
        (project / "review").chmod(0o755)
        (project / "config.toml").chmod(0o644)
        return venv, project

    @staticmethod
    def _launcher_paths(project: Path) -> tuple[Path, ...]:
        return tuple(
            project / basename
            for basename in (
                installer.REVIEW_SHELL_BASENAME,
                installer.REVIEW_DESKTOP_BASENAME,
                installer.ROUND_REVIEW_SHELL_BASENAME,
                installer.ROUND_REVIEW_DESKTOP_BASENAME,
            )
        )

    @staticmethod
    def _round_inputs(root: Path) -> tuple[Path, Path, Path]:
        selection = root / "round selection operation"
        selection_payload = selection / "payload/selection"
        selection_payload.mkdir(parents=True)
        (selection / "FINAL_STATUS.json").write_text("{}\n", encoding="ascii")
        (selection / "OUTPUT_MANIFEST.json").write_text("{}\n", encoding="ascii")
        (selection_payload / "review_request.csv").write_text("fixture\n", encoding="ascii")
        (selection_payload / "round_manifest.json").write_text("{}\n", encoding="ascii")

        stage60 = root / "sealed stage 60"
        stage60.mkdir()
        (stage60 / "predictions.csv").write_text("fixture\n", encoding="ascii")
        (stage60 / "raw_features.csv").write_text("fixture\n", encoding="ascii")
        (stage60 / "inference_result.json").write_text("{}\n", encoding="ascii")

        images = root / "original inference images"
        images.mkdir()
        (images / "image one.jpg").write_bytes(b"fixture\n")
        return selection, stage60, images

    @staticmethod
    def _tree_digest(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(relative + b"\0")
            info = path.lstat()
            digest.update(f"{stat.S_IFMT(info.st_mode):o}:{stat.S_IMODE(info.st_mode):o}".encode("ascii"))
            if stat.S_ISREG(info.st_mode):
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def test_installs_executable_no_clobber_launchers_for_lite_and_full(self) -> None:
        for profile in installer.PRESET_PROFILES.values():
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                venv, project = self._fixture(Path(directory), profile)
                result = installer.install_review_launchers(venv=venv, project=project)
                self.assertEqual(result["schema"], installer.REVIEW_LAUNCHER_SCHEMA)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["profile"], profile)
                self.assertEqual(result["venv_python"], str(venv / "bin/python"))
                self.assertEqual(result["published_project"], str(project))

                shell = project / installer.REVIEW_SHELL_BASENAME
                desktop = project / installer.REVIEW_DESKTOP_BASENAME
                round_shell = project / installer.ROUND_REVIEW_SHELL_BASENAME
                round_desktop = project / installer.ROUND_REVIEW_DESKTOP_BASENAME
                for path in self._launcher_paths(project):
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
                    self.assertEqual(path.stat().st_nlink, 1)
                shell_text = shell.read_text(encoding="utf-8")
                desktop_text = desktop.read_text(encoding="utf-8")
                round_shell_text = round_shell.read_text(encoding="utf-8")
                round_desktop_text = round_desktop.read_text(encoding="utf-8")
                self.assertIn(f"COMPAG_PYTHON='{venv / 'bin/python'}'", shell_text)
                self.assertIn('review-ui "$REVIEW_UI_MODE"', shell_text)
                self.assertIn('--target-label "$TARGET_LABEL"', shell_text)
                self.assertIn('--non-target-label "$NON_TARGET_LABEL"', shell_text)
                self.assertIn('--model-preset "$MODEL_PRESET"', shell_text)
                self.assertIn("compag-cj-r92", shell_text)
                self.assertIn("review/cj-r92-reviewed.csv", shell_text)
                self.assertIn("threshold 0.50", shell_text)
                self.assertIn("not fresh canonical equivalence", shell_text)
                self.assertIn('"$ROUND_REVIEW_LAUNCHER"', shell_text)
                self.assertNotIn("eval ", shell_text)
                self.assertIn("Terminal=true", desktop_text)
                self.assertIn(f'Exec="{shell}"', desktop_text)
                self.assertIn("Master Launcher", desktop_text)
                self.assertIn(f"COMPAG_PYTHON='{venv / 'bin/python'}'", round_shell_text)
                self.assertIn('--selection-root "$SELECTION_ROOT"', round_shell_text)
                self.assertIn('--stage60-root "$STAGE60_ROOT"', round_shell_text)
                self.assertIn('--images "$IMAGES_PATH"', round_shell_text)
                self.assertIn('--output "$OUTPUT_PATH"', round_shell_text)
                self.assertIn('review-ui "$REVIEW_UI_MODE"', round_shell_text)
                self.assertIn('--target-label "$TARGET_LABEL"', round_shell_text)
                self.assertIn('--non-target-label "$NON_TARGET_LABEL"', round_shell_text)
                self.assertNotIn("eval ", round_shell_text)
                self.assertNotIn("resume-round", round_shell_text)
                self.assertIn("Terminal=true", round_desktop_text)
                self.assertIn(f'Exec="{round_shell}"', round_desktop_text)
                self.assertFalse(any(project.glob(".compag-review-launcher.*")))
                self.assertEqual(len(result["launchers"]), 4)
                self.assertEqual(
                    {row["path"] for row in result["launchers"]},
                    {str(path) for path in self._launcher_paths(project)},
                )
                for row in result["launchers"]:
                    payload = Path(row["path"]).read_bytes()
                    self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())
                    self.assertEqual(row["mode"], "0755")

    def test_attested_conda_0775_python_is_hardened_before_launchers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = installer.PRESET_PROFILES["full"]
            venv, project = self._group_writable_conda_fixture(root, profile)
            target = venv / "bin/python3.12"

            result = installer.install_review_launchers(venv=venv, project=project)

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["venv_python"], str(venv / "bin/python"))
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
            for launcher in self._launcher_paths(project):
                self.assertTrue(launcher.is_file())

    def test_fresh_environment_hardening_uses_fd_and_does_not_require_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, _project = self._group_writable_conda_fixture(
                root,
                installer.PRESET_PROFILES["lite"],
            )
            (venv / installer.RECEIPT_RELATIVE).unlink()
            target = venv / "bin/python3.12"
            actual_fchmod = os.fchmod
            calls: list[tuple[int, int]] = []

            def record_fchmod(descriptor: int, mode: int) -> None:
                calls.append((descriptor, mode))
                actual_fchmod(descriptor, mode)

            with mock.patch.object(installer.os, "fchmod", side_effect=record_fchmod):
                result = installer._verified_venv_python(venv, None)

            self.assertEqual(result, venv / "bin/python")
            self.assertEqual(len(calls), 1)
            self.assertGreaterEqual(calls[0][0], 0)
            self.assertEqual(calls[0][1], 0o755)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)

    def test_verify_venv_python_cli_attests_receipt_metadata_and_repairs_0775(self) -> None:
        for initial_mode in (0o755, 0o775):
            with self.subTest(initial_mode=oct(initial_mode)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                venv, _project = self._group_writable_conda_fixture(
                    root,
                    installer.PRESET_PROFILES["full"],
                )
                target = venv / "bin/python3.12"
                target.chmod(initial_mode)
                output = io.StringIO()

                with contextlib.redirect_stdout(output):
                    code = installer.main(["--verify-venv-python", "--venv", str(venv)])

                self.assertEqual(code, 0)
                self.assertEqual(output.getvalue().count("\n"), 1)
                self.assertEqual(
                    json.loads(output.getvalue()),
                    {
                        "schema": installer.VENV_PYTHON_VERIFICATION_SCHEMA,
                        "status": "PASS",
                        "venv_python": str(venv / "bin/python"),
                    },
                )
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)

    def test_verify_venv_python_cli_rejects_unrelated_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._group_writable_conda_fixture(
                root,
                installer.PRESET_PROFILES["lite"],
            )
            error = io.StringIO()

            with contextlib.redirect_stderr(error):
                code = installer.main(
                    [
                        "--verify-venv-python",
                        "--venv",
                        str(venv),
                        "--project",
                        str(project),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("accepts only", error.getvalue())
            self.assertEqual(stat.S_IMODE((venv / "bin/python3.12").stat().st_mode), 0o775)

    def test_verify_venv_python_cli_ignores_false_mamba_in_prefix_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, _project = self._group_writable_conda_fixture(
                root,
                installer.PRESET_PROFILES["full"],
            )
            target = venv / "bin/python3.12"
            target.chmod(0o755)
            metadata = next((venv / "conda-meta").glob("python-3.12.7-*.json"))
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["paths_data"]["paths"][0]["sha256_in_prefix"] = "d" * 64
            metadata.write_text(
                json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata.chmod(0o644)
            error = io.StringIO()

            with contextlib.redirect_stderr(error):
                code = installer.main(["--verify-venv-python", "--venv", str(venv)])

            self.assertEqual(code, 0)
            self.assertEqual(error.getvalue(), "")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)

    def test_relocation_attestation_rejects_payload_tamper_and_noncanonical_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "exact conda environment"
            encoded_prefix = os.fsencode(venv)
            placeholder = installer.CONDA_PYTHON_PREFIX_PLACEHOLDER
            relocated = encoded_prefix + b"\0" * (len(placeholder) - len(encoded_prefix))
            installed = self._conda_python_source.replace(placeholder, relocated, 1)

            installer._attest_conda_python_relocation(
                installed,
                venv,
                installer.CONDA_PYTHON_BINARY_SOURCE_SHA256,
            )

            tampered = bytearray(installed)
            tampered[0] ^= 1
            with self.assertRaisesRegex(installer.InstallerError, "digest does not match"):
                installer._attest_conda_python_relocation(
                    bytes(tampered),
                    venv,
                    installer.CONDA_PYTHON_BINARY_SOURCE_SHA256,
                )

            with self.assertRaisesRegex(installer.InstallerError, "exact locked prefix relocation"):
                installer._attest_conda_python_relocation(
                    installed + relocated,
                    venv,
                    installer.CONDA_PYTHON_BINARY_SOURCE_SHA256,
                )

            wrong_prefix = bytearray(installed)
            start = installed.index(relocated)
            wrong_prefix[start] ^= 1
            with self.assertRaisesRegex(installer.InstallerError, "exact locked prefix relocation"):
                installer._attest_conda_python_relocation(
                    bytes(wrong_prefix),
                    venv,
                    installer.CONDA_PYTHON_BINARY_SOURCE_SHA256,
                )

    def test_python_attestation_rejects_locked_package_identity_tamper(self) -> None:
        mutations = {
            "archive": ("sha256", "0" * 64),
            "source": ("paths_data.paths.sha256", "0" * 64),
            "in-prefix-format": ("paths_data.paths.sha256_in_prefix", "not-a-digest"),
        }
        for mutation, (field, replacement) in mutations.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                venv, project = self._group_writable_conda_fixture(
                    root,
                    installer.PRESET_PROFILES["full"],
                )
                metadata = next((venv / "conda-meta").glob("python-3.12.7-*.json"))
                value = json.loads(metadata.read_text(encoding="utf-8"))
                if field == "sha256":
                    value[field] = replacement
                else:
                    value["paths_data"]["paths"][0][field.rsplit(".", 1)[-1]] = replacement
                metadata.write_text(
                    json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                metadata.chmod(0o644)

                with self.assertRaisesRegex(installer.InstallerError, "identity is invalid"):
                    installer.install_review_launchers(venv=venv, project=project)

                self.assertEqual(stat.S_IMODE((venv / "bin/python3.12").stat().st_mode), 0o775)
                for launcher in self._launcher_paths(project):
                    self.assertFalse(launcher.exists())

    def test_group_writable_conda_python_requires_exact_receipt_before_launcher_hardening(self) -> None:
        mutations = {
            "schema": "wrong-schema/v1",
            "release_version": "9.9.9",
            "profile": installer.PRESET_PROFILES["lite"],
            "venv": "/tmp/wrong-environment",
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile = installer.PRESET_PROFILES["full"]
                venv, project = self._group_writable_conda_fixture(root, profile)
                receipt_path = venv / installer.RECEIPT_RELATIVE
                receipt = json.loads(receipt_path.read_text(encoding="ascii"))
                receipt[field] = replacement
                receipt_path.write_text(
                    json.dumps(receipt, ensure_ascii=True, sort_keys=True) + "\n",
                    encoding="ascii",
                )
                receipt_path.chmod(0o644)

                with self.assertRaisesRegex(installer.InstallerError, "does not attest"):
                    installer.install_review_launchers(venv=venv, project=project)

                self.assertEqual(stat.S_IMODE((venv / "bin/python3.12").stat().st_mode), 0o775)
                for launcher in self._launcher_paths(project):
                    self.assertFalse(launcher.exists())

    def test_group_writable_conda_python_rejects_digest_or_metadata_permissions(self) -> None:
        for mutation in ("digest", "metadata-mode"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile = installer.PRESET_PROFILES["full"]
                venv, project = self._group_writable_conda_fixture(root, profile)
                metadata = next((venv / "conda-meta").glob("python-3.12.7-*.json"))
                if mutation == "digest":
                    binary = venv / "bin/python3.12"
                    payload = bytearray(binary.read_bytes())
                    payload[0] ^= 1
                    binary.write_bytes(payload)
                    binary.chmod(0o775)
                    message = "digest does not match"
                else:
                    metadata.chmod(0o664)
                    message = "mode-0644"

                with self.assertRaisesRegex(installer.InstallerError, message):
                    installer.install_review_launchers(venv=venv, project=project)

                self.assertEqual(stat.S_IMODE((venv / "bin/python3.12").stat().st_mode), 0o775)
                for launcher in self._launcher_paths(project):
                    self.assertFalse(launcher.exists())

    def test_other_writable_conda_python_is_never_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._group_writable_conda_fixture(
                root,
                installer.PRESET_PROFILES["full"],
            )
            target = venv / "bin/python3.12"
            target.chmod(0o777)

            with self.assertRaisesRegex(installer.InstallerError, "unsafe metadata"):
                installer.install_review_launchers(venv=venv, project=project)

            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o777)
            for launcher in self._launcher_paths(project):
                self.assertFalse(launcher.exists())

    def test_staging_install_embeds_the_absent_published_project_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, staging = self._fixture(
                root,
                installer.PRESET_PROFILES["full"],
            )
            published = root / "final published project"
            result = installer.install_review_launchers(
                venv=venv,
                project=staging,
                published_project=published,
            )
            shell = staging / installer.REVIEW_SHELL_BASENAME
            desktop = staging / installer.REVIEW_DESKTOP_BASENAME
            round_shell = staging / installer.ROUND_REVIEW_SHELL_BASENAME
            round_desktop = staging / installer.ROUND_REVIEW_DESKTOP_BASENAME
            self.assertEqual(result["project"], str(staging))
            self.assertEqual(result["published_project"], str(published))
            self.assertIn(f"COMPAG_PROJECT='{published}'", shell.read_text(encoding="utf-8"))
            self.assertIn(
                f'Exec="{published / installer.REVIEW_SHELL_BASENAME}"',
                desktop.read_text(encoding="utf-8"),
            )
            self.assertIn(f"COMPAG_PROJECT='{published}'", round_shell.read_text(encoding="utf-8"))
            self.assertIn(
                f'Exec="{published / installer.ROUND_REVIEW_SHELL_BASENAME}"',
                round_desktop.read_text(encoding="utf-8"),
            )
            for path in self._launcher_paths(staging):
                self.assertNotIn(str(staging), path.read_text(encoding="utf-8"))
            self.assertFalse(published.exists())

    def test_existing_regular_or_broken_symlink_is_never_replaced(self) -> None:
        basenames = (
            installer.REVIEW_SHELL_BASENAME,
            installer.REVIEW_DESKTOP_BASENAME,
            installer.ROUND_REVIEW_SHELL_BASENAME,
            installer.ROUND_REVIEW_DESKTOP_BASENAME,
        )
        for basename in basenames:
            with self.subTest(basename=basename), tempfile.TemporaryDirectory() as directory:
                venv, project = self._fixture(Path(directory), installer.PRESET_PROFILES["lite"])
                target = project / basename
                target.write_bytes(b"user-owned launcher\n")
                before = target.read_bytes()
                with self.assertRaisesRegex(installer.InstallerError, "already exists"):
                    installer.install_review_launchers(venv=venv, project=project)
                self.assertEqual(target.read_bytes(), before)
                for path in self._launcher_paths(project):
                    if path != target:
                        self.assertFalse(path.exists())

        with tempfile.TemporaryDirectory() as directory:
            venv, project = self._fixture(Path(directory), installer.PRESET_PROFILES["full"])
            desktop = project / installer.ROUND_REVIEW_DESKTOP_BASENAME
            desktop.symlink_to(project / "missing-user-target")
            with self.assertRaisesRegex(installer.InstallerError, "already exists"):
                installer.install_review_launchers(venv=venv, project=project)
            self.assertTrue(desktop.is_symlink())
            for path in self._launcher_paths(project):
                if path != desktop:
                    self.assertFalse(path.exists())

    def test_third_publish_failure_rolls_back_and_does_not_touch_source_or_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["full"])
            source = root / "source"
            paused_run = root / "paused-run"
            source.mkdir()
            paused_run.mkdir()
            (source / "original.py").write_bytes(b"immutable source fixture\n")
            (paused_run / "RUN_OWNER.json").write_bytes(b"immutable run fixture\n")
            source_before = self._tree_digest(source)
            run_before = self._tree_digest(paused_run)
            actual_link = os.link
            calls = 0

            def fail_third_link(src: Path, dst: Path, *, follow_symlinks: bool) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected third publication failure")
                actual_link(src, dst, follow_symlinks=follow_symlinks)

            with mock.patch.object(installer.os, "link", side_effect=fail_third_link):
                with self.assertRaisesRegex(installer.InstallerError, "failed without replacing"):
                    installer.install_review_launchers(venv=venv, project=project)

            self.assertEqual(source_before, self._tree_digest(source))
            self.assertEqual(run_before, self._tree_digest(paused_run))
            for path in self._launcher_paths(project):
                self.assertFalse(path.exists())
            self.assertFalse(any(project.glob(".compag-review-launcher.*")))

    def test_post_commit_cleanup_failure_keeps_verified_install_successful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["full"])
            actual_unlink = pathlib.Path.unlink

            def fail_private_stage_unlink(path: Path, *args: object, **kwargs: object) -> None:
                if path.parent.name.startswith(".compag-review-launcher."):
                    raise OSError(5, "injected persistent staging cleanup failure")
                actual_unlink(path, *args, **kwargs)

            error = io.StringIO()
            with mock.patch.object(pathlib.Path, "unlink", fail_private_stage_unlink):
                with contextlib.redirect_stderr(error):
                    result = installer.install_review_launchers(venv=venv, project=project)

            self.assertEqual(result["status"], "PASS")
            self.assertIn("committed and verified", error.getvalue())
            stages = tuple(project.glob(".compag-review-launcher.*"))
            self.assertEqual(len(stages), 1)
            self.assertEqual(stat.S_IMODE(stages[0].stat().st_mode), 0o700)
            for row, path in zip(result["launchers"], self._launcher_paths(project), strict=True):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
                self.assertEqual(path.stat().st_nlink, 2)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_transient_post_commit_cleanup_failure_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["lite"])
            actual_unlink = pathlib.Path.unlink
            injected = False

            def fail_once(path: Path, *args: object, **kwargs: object) -> None:
                nonlocal injected
                if path.parent.name.startswith(".compag-review-launcher.") and not injected:
                    injected = True
                    raise OSError(5, "injected transient staging cleanup failure")
                actual_unlink(path, *args, **kwargs)

            error = io.StringIO()
            with mock.patch.object(pathlib.Path, "unlink", fail_once):
                with contextlib.redirect_stderr(error):
                    result = installer.install_review_launchers(venv=venv, project=project)

            self.assertTrue(injected)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(error.getvalue(), "")
            self.assertFalse(any(project.glob(".compag-review-launcher.*")))
            for path in self._launcher_paths(project):
                self.assertEqual(path.stat().st_nlink, 1)

    def test_ubuntu_launcher_paths_reject_control_characters_and_backslashes(self) -> None:
        for component, message in (
            ("tab\tcomponent", "control character"),
            ("back\\slash", "backslash"),
        ):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                unsafe_root = Path(directory) / component
                unsafe_root.mkdir()
                venv, project = self._fixture(unsafe_root, installer.PRESET_PROFILES["full"])
                with self.assertRaisesRegex(installer.InstallerError, message):
                    installer.install_review_launchers(venv=venv, project=project)
                for path in self._launcher_paths(project):
                    self.assertFalse(path.exists())

    def test_master_project1_uses_fixed_cj_preset_threshold_wording_and_distinct_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["full"])
            argument_log = root / "project1-arguments.bin"
            python = venv / "bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\0' \"$@\" > {argument_log!s}\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            installer.install_review_launchers(venv=venv, project=project)

            paused_run = root / "Project 1 paused run"
            review_stage = paused_run / "stages/20_proposals_features"
            review_stage.mkdir(parents=True)
            (paused_run / "RUN_OWNER.json").write_text("{}\n", encoding="ascii")
            (review_stage / "review_request.csv").write_text("fixture\n", encoding="ascii")
            default_reviewed = project / "review/cj-r92-reviewed.csv"

            completed = subprocess.run(
                [str(project / installer.REVIEW_SHELL_BASENAME)],
                input=f"1\n2\n{paused_run}\n\n",
                check=False,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = [
                value.decode("utf-8")
                for value in argument_log.read_bytes().split(b"\0")[:-1]
            ]
            self.assertEqual(
                arguments,
                [
                    "-I", "-B", "-m", "compag_curation", "review-ui", "web",
                    "--run", str(paused_run),
                    "--output", str(default_reviewed),
                    "--target-label", "CJ",
                    "--non-target-label", "Non-CJ",
                    "--model-preset", "compag-cj-r92",
                ],
            )
            self.assertIn("fixed at 0.50", completed.stdout)
            self.assertIn("does not claim fresh canonical training or equivalence", completed.stdout)
            self.assertNotIn("Choose the project label vocabulary", completed.stdout)
            self.assertFalse(default_reviewed.exists())

    def test_shell_prompts_and_invokes_only_exact_venv_python_with_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["lite"])
            argument_log = root / "arguments.bin"
            python = venv / "bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\0' \"$@\" > {argument_log!s}\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            installer.install_review_launchers(venv=venv, project=project)

            paused_run = root / "paused run"
            review_stage = paused_run / "stages/20_proposals_features"
            review_stage.mkdir(parents=True)
            (paused_run / "RUN_OWNER.json").write_text("{}\n", encoding="ascii")
            (review_stage / "review_request.csv").write_text("fixture\n", encoding="ascii")
            reviewed = project / "review/final reviewed.csv"
            shell = project / installer.REVIEW_SHELL_BASENAME
            completed = subprocess.run(
                [str(shell)],
                input=f"2\n1\n1\n{paused_run}\n{reviewed}\n",
                check=False,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = argument_log.read_bytes().split(b"\0")[:-1]
            self.assertEqual(
                [value.decode("utf-8") for value in arguments],
                [
                    "-I", "-B", "-m", "compag_curation", "review-ui", "desktop",
                    "--run", str(paused_run), "--output", str(reviewed),
                    "--target-label", "CJ", "--non-target-label", "Non-CJ",
                ],
            )
            self.assertFalse(reviewed.exists())

    def test_custom_labels_are_literal_argv_and_cannot_inject_shell_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["lite"])
            argument_log = root / "custom-label-arguments.bin"
            python = venv / "bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\0' \"$@\" > {argument_log!s}\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            installer.install_review_launchers(venv=venv, project=project)

            paused_run = root / "paused run"
            review_stage = paused_run / "stages/20_proposals_features"
            review_stage.mkdir(parents=True)
            (paused_run / "RUN_OWNER.json").write_text("{}\n", encoding="ascii")
            (review_stage / "review_request.csv").write_text("fixture\n", encoding="ascii")
            reviewed = project / "review/custom reviewed.csv"
            injected = root / "PWNED_BY_LABEL"
            target_label = 'CJ $(touch PWNED_BY_LABEL); "$HOME"'
            non_target_label = r"Not CJ \\ literal"

            completed = subprocess.run(
                [str(project / installer.REVIEW_SHELL_BASENAME)],
                input=(
                    f"2\n2\n3\n{target_label}\n{non_target_label}\n"
                    f"{paused_run}\n{reviewed}\n"
                ),
                cwd=root,
                check=False,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(injected.exists())
            arguments = [
                value.decode("utf-8")
                for value in argument_log.read_bytes().split(b"\0")[:-1]
            ]
            self.assertEqual(
                arguments,
                [
                    "-I", "-B", "-m", "compag_curation", "review-ui", "web",
                    "--run", str(paused_run), "--output", str(reviewed),
                    "--target-label", target_label,
                    "--non-target-label", non_target_label,
                ],
            )

    def test_both_launchers_reject_unsafe_custom_label_text_before_python(self) -> None:
        cases = (
            (" Target", "Other", "surrounding whitespace"),
            ("Target ", "Other", "surrounding whitespace"),
            ("Tar\tget", "Other", "control characters"),
            ("Tar\rget", "Other", "control characters"),
            ("A" * 41, "Other", "at most 40"),
            ("CJ", "cj", "must differ"),
        )
        for launcher_basename in (
            installer.REVIEW_SHELL_BASENAME,
            installer.ROUND_REVIEW_SHELL_BASENAME,
        ):
            for target_label, non_target_label, message in cases:
                with (
                    self.subTest(
                        launcher=launcher_basename,
                        target_label=repr(target_label),
                    ),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    venv, project = self._fixture(root, installer.PRESET_PROFILES["full"])
                    invoked = root / "python-was-invoked"
                    python = venv / "bin/python"
                    python.write_text(
                        "#!/usr/bin/env bash\n"
                        f"touch {invoked!s}\n",
                        encoding="utf-8",
                    )
                    python.chmod(0o755)
                    installer.install_review_launchers(venv=venv, project=project)

                    workflow = "2\n" if launcher_basename == installer.REVIEW_SHELL_BASENAME else ""
                    completed = subprocess.run(
                        [str(project / launcher_basename)],
                        input=f"{workflow}1\n3\n{target_label}\n{non_target_label}\n",
                        check=False,
                        shell=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    self.assertEqual(completed.returncode, 2)
                    self.assertIn(message, completed.stderr)
                    self.assertFalse(invoked.exists())

    def test_primary_model_choice_delegates_to_round_chooser_with_exact_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["full"])
            argument_log = root / "delegated-round-arguments.bin"
            python = venv / "bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\0' \"$@\" > {argument_log!s}\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            installer.install_review_launchers(venv=venv, project=project)
            selection, stage60, images = self._round_inputs(root)
            reviewed = project / "review/delegated reviewed.csv"

            completed = subprocess.run(
                [str(project / installer.REVIEW_SHELL_BASENAME)],
                input=(
                    f"3\n2\n1\n{selection}\n{stage60}\n{images}\n{reviewed}\n"
                ),
                check=False,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = [
                value.decode("utf-8")
                for value in argument_log.read_bytes().split(b"\0")[:-1]
            ]
            self.assertEqual(
                arguments,
                [
                    "-I", "-B", "-m", "compag_curation", "review-ui", "web",
                    "--selection-root", str(selection),
                    "--stage60-root", str(stage60),
                    "--images", str(images),
                    "--output", str(reviewed),
                    "--target-label", "CJ", "--non-target-label", "Non-CJ",
                ],
            )
            self.assertNotIn("resume-round", arguments)

    def test_launchers_reject_symlink_traversal_before_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["lite"])
            invoked = root / "python-was-invoked"
            python = venv / "bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {invoked!s}\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            installer.install_review_launchers(venv=venv, project=project)

            physical_parent = root / "physical"
            paused_run = physical_parent / "paused-run"
            review_stage = paused_run / "stages/20_proposals_features"
            review_stage.mkdir(parents=True)
            (paused_run / "RUN_OWNER.json").write_text("{}\n", encoding="ascii")
            (review_stage / "review_request.csv").write_text("fixture\n", encoding="ascii")
            alias = root / "alias"
            alias.symlink_to(physical_parent, target_is_directory=True)
            traversing_run = alias / "paused-run"

            completed = subprocess.run(
                [str(project / installer.REVIEW_SHELL_BASENAME)],
                input=f"2\n1\n1\n{traversing_run}\n",
                check=False,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("without symlink traversal", completed.stderr)
            self.assertFalse(invoked.exists())

    def test_round_shell_preserves_spaced_paths_as_exact_argv_and_does_not_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["full"])
            argument_log = root / "round-arguments.bin"
            python = venv / "bin/python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\0' \"$@\" > {argument_log!s}\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            installer.install_review_launchers(venv=venv, project=project)
            selection, stage60, images = self._round_inputs(root)
            reviewed = project / "review/round one reviewed.csv"

            completed = subprocess.run(
                [str(project / installer.ROUND_REVIEW_SHELL_BASENAME)],
                input=f"2\n2\n{selection}\n{stage60}\n{images}\n{reviewed}\n",
                check=False,
                shell=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = argument_log.read_bytes().split(b"\0")[:-1]
            self.assertEqual(
                [value.decode("utf-8") for value in arguments],
                [
                    "-I", "-B", "-m", "compag_curation", "review-ui", "web",
                    "--selection-root", str(selection),
                    "--stage60-root", str(stage60),
                    "--images", str(images),
                    "--output", str(reviewed),
                    "--target-label", "Target", "--non-target-label", "Non-target",
                ],
            )
            self.assertNotIn("resume-round", [value.decode("utf-8") for value in arguments])
            self.assertFalse(reviewed.exists())

    def test_round_shell_rejects_output_within_each_immutable_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv, project = self._fixture(root, installer.PRESET_PROFILES["lite"])
            installer.install_review_launchers(venv=venv, project=project)
            selection, stage60, images = self._round_inputs(root)
            shell = project / installer.ROUND_REVIEW_SHELL_BASENAME

            for immutable in (selection, stage60, images):
                with self.subTest(immutable=immutable):
                    reviewed = immutable / "must-not-be-written.csv"
                    completed = subprocess.run(
                        [str(shell)],
                        input=f"1\n1\n{selection}\n{stage60}\n{images}\n{reviewed}\n",
                        check=False,
                        shell=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn("outside all three immutable inputs", completed.stderr)
                    self.assertFalse(reviewed.exists())

    def test_cli_mode_emits_one_json_line_and_rejects_unrelated_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            venv, project = self._fixture(Path(directory), installer.PRESET_PROFILES["full"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = installer.main(
                    ["--install-review-launcher", "--venv", str(venv), "--project", str(project)]
                )
            self.assertEqual(code, 0)
            self.assertEqual(output.getvalue().count("\n"), 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "PASS")

        with tempfile.TemporaryDirectory() as directory:
            venv, project = self._fixture(Path(directory), installer.PRESET_PROFILES["lite"])
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = installer.main(
                    [
                        "--install-review-launcher", "--venv", str(venv),
                        "--project", str(project), "--execute",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("accepts only", error.getvalue())
            self.assertFalse((project / installer.REVIEW_SHELL_BASENAME).exists())


if __name__ == "__main__":
    unittest.main()
