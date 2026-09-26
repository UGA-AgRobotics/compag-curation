from __future__ import annotations

import contextlib
import errno
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest import mock

from tools.post_upload_metadata_finalizer import FinalizationError, finalize, main as deprecated_main
from tools.release_provenance_verifier import (
    ProvenanceError,
    validate_identity,
    verify_release_provenance,
)


REPOSITORY_URL = "https://github.com/acme-labs/compag-curation"
TAG = "v1.9.3"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class FakeGitRunner:
    """In-process Git fixture used only inside the guarded no-process QA run."""

    def __init__(self, repository: Path) -> None:
        self.repository = repository.resolve()
        self.commit = FAKE_COMMIT
        self.head = self.commit
        self.tag_commit = self.commit
        self.remote_url = REPOSITORY_URL
        self.tracked: dict[str, tuple[str, bytes]] = {}
        for relative in ("pyproject.toml", "src/run-tool", "src/payload.txt"):
            path = self.repository / relative
            mode = "100755" if stat.S_IMODE(path.stat().st_mode) & 0o111 else "100644"
            self.tracked[relative] = (mode, path.read_bytes())
        self.objects = {
            hashlib.sha1(payload).hexdigest(): payload
            for _relative, (_mode, payload) in self.tracked.items()
        }

    def _dirty(self) -> bool:
        for relative, (mode, payload) in self.tracked.items():
            path = self.repository / relative
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                return True
            observed_mode = "100755" if stat.S_IMODE(path.stat().st_mode) & 0o111 else "100644"
            if observed_mode != mode:
                return True
        return False

    def __call__(self, _git: str, repository: Path, arguments: Sequence[str]) -> bytes:
        if repository.resolve() != self.repository:
            raise AssertionError("fake Git runner received the wrong repository root")
        command = tuple(arguments)
        if command == ("rev-parse", "--show-toplevel"):
            return f"{self.repository}\n".encode()
        if command == ("rev-parse", "--is-bare-repository"):
            return b"false\n"
        if command == ("rev-parse", "--verify", "HEAD"):
            return f"{self.head}\n".encode()
        if command == ("rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"):
            return f"{self.tag_commit}\n".encode()
        if command == ("remote", "get-url", "origin"):
            return f"{self.remote_url}\n".encode()
        if command == ("status", "--porcelain=v1", "-z", "--untracked-files=no"):
            return b" M src/payload.txt\0" if self._dirty() else b""
        if command == ("ls-files", "--stage", "-z"):
            rows = []
            for relative, (mode, payload) in sorted(self.tracked.items()):
                object_id = hashlib.sha1(payload).hexdigest()
                rows.append(f"{mode} {object_id} 0\t{relative}".encode())
            return b"\0".join(rows) + b"\0"
        if command == ("show", f"{self.commit}:pyproject.toml"):
            return self.tracked["pyproject.toml"][1]
        if command == ("ls-tree", "-r", "-z", "-l", "--full-tree", self.commit):
            rows = []
            for relative, (mode, payload) in sorted(self.tracked.items()):
                object_id = hashlib.sha1(payload).hexdigest()
                rows.append(f"{mode} blob {object_id} {len(payload)}\t{relative}".encode())
            return b"\0".join(rows) + b"\0"
        if len(command) == 3 and command[:2] == ("cat-file", "blob") and command[2] in self.objects:
            return self.objects[command[2]]
        raise AssertionError(f"unexpected fake Git command: {command!r}")


class PostUploadMetadataFinalizerTests(unittest.TestCase):
    def _git(self, repository: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    def _release(self, parent: Path) -> tuple[Path, Path, str, FakeGitRunner | None]:
        repository = parent / "repository"
        extracted = parent / "extracted-source"
        repository.mkdir()
        (repository / "src").mkdir()
        (repository / "pyproject.toml").write_text(
            """[build-system]
requires = ["setuptools==80.9.0"]
build-backend = "setuptools.build_meta"

[project]
name = "compag-curation"
version = "1.9.3"
""",
            encoding="utf-8",
        )
        executable = repository / "src" / "run-tool"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        (repository / "src" / "payload.txt").write_text("reviewed payload\n", encoding="utf-8")
        guarded = bool(os.environ.get("COMPAG_PUBLIC_ROOT"))
        runner: FakeGitRunner | None
        if guarded:
            runner = FakeGitRunner(repository)
            commit = runner.commit
        else:
            runner = None
            self._git(repository, "init", "-q")
            self._git(repository, "config", "user.name", "Release Test")
            self._git(repository, "config", "user.email", "release-test@example.invalid")
            self._git(repository, "remote", "add", "origin", REPOSITORY_URL)
            self._git(repository, "add", "--all")
            self._git(repository, "commit", "-q", "-m", "release fixture")
            commit = self._git(repository, "rev-parse", "HEAD")
            self._git(repository, "tag", "-a", TAG, "-m", "version 1.9.3")
        extracted.mkdir()
        for relative in ("pyproject.toml", "src/run-tool", "src/payload.txt"):
            destination = extracted / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repository / relative, destination)
        return repository, extracted, commit, runner

    def _verify(
        self,
        repository: Path,
        extracted: Path,
        output: Path,
        repository_url: str,
        tag: str,
        commit: str,
        runner: FakeGitRunner | None,
    ) -> dict[str, object]:
        kwargs = {"git_runner": runner} if runner is not None else {}
        return verify_release_provenance(
            repository,
            extracted,
            output,
            repository_url,
            tag,
            commit,
            **kwargs,
        )

    def test_read_only_verifier_emits_external_deterministic_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            output = parent / "evidence" / "RELEASE_PROVENANCE.json"
            output.parent.mkdir()
            repository_before = {
                relative: (repository / relative).read_bytes()
                for relative in ("pyproject.toml", "src/run-tool", "src/payload.txt")
            }
            extracted_before = {
                relative: (extracted / relative).read_bytes()
                for relative in ("pyproject.toml", "src/run-tool", "src/payload.txt")
            }
            metadata_before = {
                (label, relative): (
                    stat.S_IMODE((root / relative).stat().st_mode),
                    (root / relative).stat().st_mtime_ns,
                    (root / relative).stat().st_size,
                )
                for label, root in (("repository", repository), ("extracted", extracted))
                for relative in ("pyproject.toml", "src/run-tool", "src/payload.txt")
            }
            status_before = (
                runner("git", repository, ["status", "--porcelain=v1", "-z", "--untracked-files=no"])
                if runner is not None
                else self._git(repository, "status", "--porcelain=v1", "--untracked-files=all")
            )

            with mock.patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.fsmonitor",
                    "GIT_CONFIG_VALUE_0": "hostile-command",
                    "GIT_DIR": os.fspath(parent / "hostile-git-dir"),
                    "GIT_WORK_TREE": os.fspath(parent / "hostile-worktree"),
                },
                clear=False,
            ):
                result = self._verify(
                    repository,
                    extracted,
                    output,
                    REPOSITORY_URL,
                    TAG,
                    commit,
                    runner,
                )

            self.assertEqual(result["status"], "PASS_POINT_IN_TIME_LOCAL_TAGGED_COMMIT_AND_EXTRACTED_SOURCE_IDENTICAL")
            self.assertEqual(result["output"], str(output))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)
            report = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(report["schema"], "compag-release-provenance/v1")
            self.assertEqual(report["release_identity"]["commit"], commit)
            self.assertEqual(report["release_identity"]["repository_url"], REPOSITORY_URL)
            self.assertEqual(report["release_identity"]["tag"], TAG)
            manifest = report["normalized_tracked_file_manifest"]
            self.assertEqual(manifest["file_count"], 3)
            self.assertEqual(
                [row["relative_path"] for row in manifest["records"]],
                ["pyproject.toml", "src/payload.txt", "src/run-tool"],
            )
            executable_row = next(row for row in manifest["records"] if row["relative_path"] == "src/run-tool")
            self.assertEqual(executable_row["git_mode"], "100755")
            expected_identity = hashlib.sha256(
                json.dumps(manifest["records"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
            self.assertEqual(manifest["identity_sha256"], expected_identity)
            self.assertEqual(result["manifest_identity_sha256"], expected_identity)
            self.assertTrue(report["side_effect_policy"]["wrote_only_external_provenance_file"])
            for relative, payload in repository_before.items():
                self.assertEqual((repository / relative).read_bytes(), payload)
            for relative, payload in extracted_before.items():
                self.assertEqual((extracted / relative).read_bytes(), payload)
            metadata_after = {
                (label, relative): (
                    stat.S_IMODE((root / relative).stat().st_mode),
                    (root / relative).stat().st_mtime_ns,
                    (root / relative).stat().st_size,
                )
                for label, root in (("repository", repository), ("extracted", extracted))
                for relative in ("pyproject.toml", "src/run-tool", "src/payload.txt")
            }
            self.assertEqual(metadata_after, metadata_before)
            if runner is not None:
                self.assertEqual(
                    runner("git", repository, ["status", "--porcelain=v1", "-z", "--untracked-files=no"]),
                    status_before,
                )
                self.assertEqual(runner.tag_commit, commit)
            else:
                self.assertEqual(
                    self._git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
                    status_before,
                )
                self.assertEqual(self._git(repository, "rev-parse", f"refs/tags/{TAG}^{{commit}}"), commit)

    def test_mismatched_missing_or_extra_extracted_content_fails_without_output(self) -> None:
        for mutation in ("changed", "missing", "extra", "mode"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                repository, extracted, commit, runner = self._release(parent)
                if mutation == "changed":
                    (extracted / "src/payload.txt").write_text("different\n", encoding="utf-8")
                elif mutation == "missing":
                    (extracted / "src/payload.txt").unlink()
                elif mutation == "mode":
                    (extracted / "src/payload.txt").chmod(0o755)
                else:
                    (extracted / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
                output = parent / "RELEASE_PROVENANCE.json"
                with self.assertRaisesRegex(ProvenanceError, "does not match the tagged commit manifest"):
                    self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
                self.assertFalse(output.exists())

    def test_tracked_worktree_change_or_moved_tag_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            (repository / "src/payload.txt").write_text("modified\n", encoding="utf-8")
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "tracked Git worktree"):
                self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            if runner is not None:
                runner.tag_commit = "89abcdef0123456789abcdef0123456789abcdef"
            else:
                (repository / "second.txt").write_text("second\n", encoding="utf-8")
                self._git(repository, "add", "second.txt")
                self._git(repository, "commit", "-q", "-m", "second commit")
                self._git(repository, "tag", "-f", TAG)
                self._git(repository, "checkout", "-q", commit)
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaises(ProvenanceError):
                self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(output.exists())

    def test_output_is_no_clobber_external_and_exactly_named(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            existing = parent / "RELEASE_PROVENANCE.json"
            existing.write_text("retain\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "no-clobber"):
                self._verify(repository, extracted, existing, REPOSITORY_URL, TAG, commit, runner)
            self.assertEqual(existing.read_text(encoding="utf-8"), "retain\n")
            wrong_name = parent / "provenance.json"
            with self.assertRaisesRegex(ProvenanceError, "exactly RELEASE_PROVENANCE.json"):
                self._verify(repository, extracted, wrong_name, REPOSITORY_URL, TAG, commit, runner)
            inside = repository / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "outside the repository"):
                self._verify(repository, extracted, inside, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(inside.exists())
            inside_extraction = extracted / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "outside the extracted source"):
                self._verify(repository, extracted, inside_extraction, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(inside_extraction.exists())
            target = parent / "retain-target"
            target.write_text("retain\n", encoding="utf-8")
            linked_output = parent / "linked-output" / "RELEASE_PROVENANCE.json"
            linked_output.parent.symlink_to(parent)
            with self.assertRaisesRegex(ProvenanceError, "output parent must be a real"):
                self._verify(repository, extracted, linked_output, REPOSITORY_URL, TAG, commit, runner)
            self.assertEqual(target.read_text(encoding="utf-8"), "retain\n")
            output_symlink = parent / "no-clobber" / "RELEASE_PROVENANCE.json"
            output_symlink.parent.mkdir()
            output_symlink.symlink_to(target)
            with self.assertRaisesRegex(ProvenanceError, "no-clobber"):
                self._verify(repository, extracted, output_symlink, REPOSITORY_URL, TAG, commit, runner)
            self.assertEqual(target.read_text(encoding="utf-8"), "retain\n")

    def test_identity_and_configured_remote_are_fail_closed(self) -> None:
        invalid = (
            ("http://github.com/acme-labs/compag-curation", TAG, "0123456789abcdef0123456789abcdef01234567"),
            ("https://gitlab.com/acme-labs/compag-curation", TAG, "0123456789abcdef0123456789abcdef01234567"),
            ("https://github.com/example/compag-curation", TAG, "0123456789abcdef0123456789abcdef01234567"),
            (REPOSITORY_URL, "main", "0123456789abcdef0123456789abcdef01234567"),
            (REPOSITORY_URL, TAG, "A" * 40),
            (REPOSITORY_URL, TAG, "0" * 40),
        )
        for repository_url, tag, commit in invalid:
            with self.subTest(repository_url=repository_url, tag=tag, commit=commit):
                with self.assertRaises(ProvenanceError):
                    validate_identity(repository_url, "1.9.3", tag, commit)

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            if runner is not None:
                runner.remote_url = "https://github.com/acme-labs/different"
            else:
                self._git(repository, "remote", "set-url", "origin", "https://github.com/acme-labs/different")
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "configured Git remote"):
                self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(output.exists())

    def test_arbitrary_absolute_git_wrapper_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, _runner = self._release(parent)
            marker = parent / "wrapper-executed"
            wrapper = parent / "git-wrapper"
            wrapper.write_text(f"#!/bin/sh\ntouch '{marker}'\nexec /usr/bin/git \"$@\"\n", encoding="utf-8")
            wrapper.chmod(0o755)
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "trusted system executable"):
                verify_release_provenance(
                    repository,
                    extracted,
                    output,
                    REPOSITORY_URL,
                    TAG,
                    commit,
                    git=os.fspath(wrapper),
                )
            self.assertFalse(marker.exists())
            self.assertFalse(output.exists())

    def test_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            (extracted / "src/payload.txt").unlink()
            (extracted / "src/payload.txt").symlink_to("run-tool")
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "symbolic link"):
                self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            (extracted / "src/payload.txt").unlink()
            os.link(repository / "src/payload.txt", extracted / "src/payload.txt")
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "single-link regular file"):
                self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(output.exists())

    def test_input_roots_must_be_real_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            repository_link = parent / "repository-link"
            repository_link.symlink_to(repository, target_is_directory=True)
            output = parent / "RELEASE_PROVENANCE.json"
            with self.assertRaisesRegex(ProvenanceError, "repository root must be a real directory"):
                self._verify(repository_link, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            extracted_link = parent / "extracted-link"
            extracted_link.symlink_to(extracted, target_is_directory=True)
            with self.assertRaisesRegex(ProvenanceError, "extracted source root must be a real directory"):
                self._verify(repository, extracted_link, output, REPOSITORY_URL, TAG, commit, runner)
            nested = repository / "nested-extracted"
            shutil.copytree(extracted, nested)
            with self.assertRaisesRegex(ProvenanceError, "must not overlap"):
                self._verify(repository, nested, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertFalse(output.exists())

    def test_git_replace_objects_cannot_rebind_the_asserted_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            if runner is None:
                (repository / "replacement-only.txt").write_text("replacement\n", encoding="utf-8")
                self._git(repository, "add", "replacement-only.txt")
                self._git(repository, "commit", "-q", "-m", "replacement object")
                replacement_commit = self._git(repository, "rev-parse", "HEAD")
                self._git(repository, "checkout", "-q", commit)
                self._git(repository, "replace", commit, replacement_commit)
            output = parent / "RELEASE_PROVENANCE.json"
            result = self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertEqual(result["status"], "PASS_POINT_IN_TIME_LOCAL_TAGGED_COMMIT_AND_EXTRACTED_SOURCE_IDENTICAL")
            report = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(report["normalized_tracked_file_manifest"]["file_count"], 3)

    def test_repository_clean_filter_is_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repository, extracted, commit, runner = self._release(parent)
            marker = parent / "clean-filter-side-effect"
            if runner is None:
                attributes = repository / ".gitattributes"
                attributes.write_text("src/payload.txt filter=evil\n", encoding="utf-8")
                self._git(repository, "add", ".gitattributes")
                self._git(repository, "commit", "-q", "-m", "tracked attributes")
                commit = self._git(repository, "rev-parse", "HEAD")
                self._git(repository, "tag", "-f", TAG)
                shutil.copy2(attributes, extracted / ".gitattributes")
                self._git(repository, "config", "filter.evil.clean", f"sh -c 'touch {marker}; cat'")
                self._git(repository, "config", "filter.evil.required", "true")
                payload = repository / "src/payload.txt"
                payload_stat = payload.stat()
                os.utime(payload, ns=(payload_stat.st_atime_ns, payload_stat.st_mtime_ns + 1_000_000_000))
            output = parent / "RELEASE_PROVENANCE.json"
            result = self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
            self.assertEqual(result["status"], "PASS_POINT_IN_TIME_LOCAL_TAGGED_COMMIT_AND_EXTRACTED_SOURCE_IDENTICAL")
            self.assertFalse(marker.exists())

    def test_publication_failures_roll_back_owned_temp_and_final_name(self) -> None:
        for failure in ("enospc", "link-race"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                repository, extracted, commit, runner = self._release(parent)
                output = parent / "RELEASE_PROVENANCE.json"
                if failure == "enospc":
                    real_write = os.write

                    def fail_write(descriptor: int, data: object) -> int:
                        view = memoryview(data)  # type: ignore[arg-type]
                        real_write(descriptor, view[: min(7, len(view))])
                        raise OSError(errno.ENOSPC, "synthetic full device")

                    patcher = mock.patch("tools.release_provenance_verifier.os.write", side_effect=fail_write)
                else:
                    patcher = mock.patch(
                        "tools.release_provenance_verifier.os.link",
                        side_effect=FileExistsError(errno.EEXIST, "synthetic publication race"),
                    )
                with patcher, self.assertRaises(ProvenanceError):
                    self._verify(repository, extracted, output, REPOSITORY_URL, TAG, commit, runner)
                self.assertFalse(output.exists())
                self.assertEqual(list(parent.glob(f".{output.name}.owned-*")), [])

    def test_retired_mutating_interface_is_explicitly_disabled(self) -> None:
        with self.assertRaisesRegex(FinalizationError, "DEPRECATED_MUTATING_FINALIZER_DISABLED"):
            finalize(Path("source"), Path("output"), REPOSITORY_URL, TAG, "0123456789abcdef0123456789abcdef01234567")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(deprecated_main(["--source-root", "never-read"]), 2)
        self.assertIn("DEPRECATED_MUTATING_FINALIZER_DISABLED", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
