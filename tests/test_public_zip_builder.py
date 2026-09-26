from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tools import build_public_zip as builder


SOURCE_DATE_EPOCH = 1_788_393_600
TOP_LEVEL = "COMPAG_D_26_02120_PUBLIC_RUNNABLE_GITHUB_v1.9.3"


class PublicZipBuilderTests(unittest.TestCase):
    def _source(self, base: Path, *, extra_paths: dict[str, bytes] | None = None) -> Path:
        root = base / "source"
        (root / "manifests").mkdir(parents=True)
        (root / "nested").mkdir()
        files = {
            "README.md": b"public source fixture\n",
            "nested/data.txt": b"deterministic payload\n",
            **(extra_paths or {}),
        }
        for relative, payload in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        allowlist_path = "manifests/public_release_allowlist.json"
        paths = sorted([*files, allowlist_path])
        allowlist = {
            "schema": "compag-curation-public-release-allowlist/v1",
            "policy": "DENY_BY_DEFAULT_EXACT_REGULAR_FILES",
            "release_version": "1.9.3",
            "path_encoding": "UTF8_NFC_POSIX_RELATIVE_SORTED_LF_TERMINATED",
            "path_count": len(paths),
            "paths_sha256": hashlib.sha256(("\n".join(paths) + "\n").encode("ascii")).hexdigest(),
            "paths": paths,
        }
        (root / allowlist_path).write_text(
            json.dumps(allowlist, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        root.chmod(0o755)
        for path in root.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        return root

    def test_two_builds_are_byte_identical_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            first = base / "first.zip"
            second = base / "second.zip"
            receipt_a = builder.build_public_zip(root, first, SOURCE_DATE_EPOCH)
            receipt_b = builder.build_public_zip(root, second, SOURCE_DATE_EPOCH)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(stat.S_IMODE(first.lstat().st_mode), 0o644)
            self.assertEqual(first.lstat().st_nlink, 1)
            self.assertEqual(receipt_a["sha256"], receipt_b["sha256"])
            self.assertEqual(receipt_a["sha256"], hashlib.sha256(first.read_bytes()).hexdigest())
            self.assertEqual(receipt_a["file_count"], 3)
            self.assertEqual(receipt_a["member_count"], 6)
            self.assertEqual(receipt_a["zip_timestamp_utc"], "2026-09-03T00:00:00Z")
            expected_names = [
                f"{TOP_LEVEL}/",
                f"{TOP_LEVEL}/README.md",
                f"{TOP_LEVEL}/manifests/",
                f"{TOP_LEVEL}/manifests/public_release_allowlist.json",
                f"{TOP_LEVEL}/nested/",
                f"{TOP_LEVEL}/nested/data.txt",
            ]
            with zipfile.ZipFile(first) as archive:
                infos = archive.infolist()
                self.assertEqual([info.filename for info in infos], expected_names)
                self.assertEqual([info.filename for info in infos], sorted(expected_names))
                self.assertEqual(archive.comment, b"")
                for info in infos:
                    self.assertEqual(info.date_time, (2026, 9, 3, 0, 0, 0))
                    self.assertEqual(info.create_system, 3)
                    self.assertEqual(info.compress_type, zipfile.ZIP_DEFLATED)
                    self.assertEqual(info.extra, b"")
                    expected_mode = (stat.S_IFDIR | 0o755) if info.is_dir() else (stat.S_IFREG | 0o644)
                    self.assertEqual(info.external_attr >> 16, expected_mode)
                self.assertEqual(archive.read(f"{TOP_LEVEL}/README.md"), b"public source fixture\n")

    def test_cli_uses_source_date_epoch_and_emits_one_compact_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            output = base / "release.zip"
            capture = io.StringIO()
            argv = ["build_public_zip.py", "--root", str(root), "--output", str(output)]
            with (
                mock.patch.object(builder.sys, "argv", argv),
                mock.patch.dict(builder.os.environ, {"SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH)}),
                contextlib.redirect_stdout(capture),
            ):
                self.assertEqual(builder.main(), 0)
            line = capture.getvalue()
            self.assertEqual(line.count("\n"), 1)
            self.assertNotIn(": ", line)
            receipt = json.loads(line)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(receipt["member_count"], 6)

    def test_exact_allowlist_and_directory_closure_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            (root / "unlisted.txt").write_bytes(b"private\n")
            (root / "unlisted.txt").chmod(0o644)
            with self.assertRaisesRegex(builder.PublicZipError, "closure mismatch"):
                builder.build_public_zip(root, base / "extra.zip", SOURCE_DATE_EPOCH)
            (root / "unlisted.txt").unlink()
            (root / "empty").mkdir(mode=0o755)
            with self.assertRaisesRegex(builder.PublicZipError, "directory closure mismatch"):
                builder.build_public_zip(root, base / "directory.zip", SOURCE_DATE_EPOCH)

    def test_no_clobber_preserves_preexisting_and_racing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            output = base / "release.zip"
            output.write_bytes(b"preexisting")
            with self.assertRaisesRegex(builder.PublicZipError, "already exists"):
                builder.build_public_zip(root, output, SOURCE_DATE_EPOCH)
            self.assertEqual(output.read_bytes(), b"preexisting")
            output.unlink()

            def racing_link(_source: object, target: object, **_kwargs: object) -> None:
                output.write_bytes(b"racing-owner")
                raise FileExistsError

            with mock.patch.object(builder.os, "link", side_effect=racing_link):
                with self.assertRaisesRegex(builder.PublicZipError, "appeared.*no-clobber"):
                    builder.build_public_zip(root, output, SOURCE_DATE_EPOCH)
            self.assertEqual(output.read_bytes(), b"racing-owner")
            self.assertEqual(list(base.glob(".release.zip.public-zip-*.tmp")), [])

    def test_post_link_foreign_replacement_is_never_adopted_or_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            output = base / "release.zip"
            original_unlink_owned = builder._unlink_owned
            swapped = False

            def swap_after_staging_unlink(
                path: Path | str | None,
                owner: tuple[int, int] | None,
                *,
                directory_fd: int | None = None,
            ) -> bool:
                nonlocal swapped
                released = original_unlink_owned(path, owner, directory_fd=directory_fd)
                if (
                    released
                    and not swapped
                    and isinstance(path, str)
                    and path.startswith(".release.zip.public-zip-")
                ):
                    swapped = True
                    output.unlink()
                    output.write_bytes(b"foreign-replacement")
                return released

            with mock.patch.object(builder, "_unlink_owned", side_effect=swap_after_staging_unlink):
                with self.assertRaisesRegex(builder.PublicZipError, "rollback was incomplete"):
                    builder.build_public_zip(root, output, SOURCE_DATE_EPOCH)
            self.assertTrue(swapped)
            self.assertEqual(output.read_bytes(), b"foreign-replacement")
            self.assertEqual(list(base.glob(".release.zip.public-zip-*.tmp")), [])

    def test_replaced_output_parent_fails_and_rolls_back_only_in_held_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            output_parent = base / "output"
            output_parent.mkdir(mode=0o755)
            moved_parent = base / "original-output"
            redirect = base / "redirect"
            redirect.mkdir(mode=0o755)
            foreign_output = redirect / "release.zip"
            foreign_output.write_bytes(b"foreign-owner")
            output = output_parent / "release.zip"
            original_write_zip = builder._write_zip
            swapped = False

            def swap_parent_after_staging_write(
                handle: io.BufferedWriter,
                rows: list[tuple[str, bool, bytes]],
                date_time: tuple[int, ...],
            ) -> None:
                nonlocal swapped
                original_write_zip(handle, rows, date_time)
                output_parent.rename(moved_parent)
                output_parent.symlink_to(redirect, target_is_directory=True)
                swapped = True

            with mock.patch.object(builder, "_write_zip", side_effect=swap_parent_after_staging_write):
                with self.assertRaisesRegex(builder.PublicZipError, "parent binding changed"):
                    builder.build_public_zip(root, output, SOURCE_DATE_EPOCH)
            self.assertTrue(swapped)
            self.assertEqual(foreign_output.read_bytes(), b"foreign-owner")
            self.assertEqual(list(redirect.glob(".release.zip.public-zip-*.tmp")), [])
            self.assertEqual(list(moved_parent.iterdir()), [])

    def test_noncanonical_prefix_and_suffix_are_rejected_and_cleaned(self) -> None:
        for placement in ("prefix", "suffix"):
            with self.subTest(placement=placement), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = self._source(base)
                output = base / "release.zip"
                original_write_zip = builder._write_zip
                hidden = b"prohibited-hidden-payload"

                def tampering_write(
                    handle: io.BufferedWriter,
                    rows: list[tuple[str, bool, bytes]],
                    date_time: tuple[int, ...],
                ) -> None:
                    if placement == "prefix":
                        handle.write(hidden)
                    original_write_zip(handle, rows, date_time)
                    if placement == "suffix":
                        handle.write(hidden)

                with mock.patch.object(builder, "_write_zip", side_effect=tampering_write):
                    with self.assertRaisesRegex(builder.PublicZipError, "canonical byte stream mismatch"):
                        builder.build_public_zip(root, output, SOURCE_DATE_EPOCH)
                self.assertFalse(output.exists())
                self.assertEqual(list(base.glob(".release.zip.public-zip-*.tmp")), [])

    def test_unsafe_output_and_top_level_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            with self.assertRaisesRegex(builder.PublicZipError, "outside the source tree"):
                builder.build_public_zip(root, root / "inside.zip", SOURCE_DATE_EPOCH)
            for index, top_level in enumerate(("../escape", "/absolute", "two/levels", ".git", "reviewer_comments")):
                with self.subTest(top_level=top_level):
                    with self.assertRaises(builder.PublicZipError):
                        builder.build_public_zip(
                            root,
                            base / f"unsafe-{index}.zip",
                            SOURCE_DATE_EPOCH,
                            top_level=top_level,
                        )
            linked_parent = base / "linked-output"
            linked_parent.symlink_to(base, target_is_directory=True)
            with self.assertRaisesRegex(builder.PublicZipError, "real non-symlink directory"):
                builder.build_public_zip(root, linked_parent / "release.zip", SOURCE_DATE_EPOCH)
            custom = base / "custom.zip"
            receipt = builder.build_public_zip(
                root,
                custom,
                SOURCE_DATE_EPOCH,
                top_level="ControlledFixtureRoot",
            )
            self.assertEqual(receipt["top_level"], "ControlledFixtureRoot")
            with zipfile.ZipFile(custom) as archive:
                self.assertEqual(
                    {Path(info.filename).parts[0] for info in archive.infolist()},
                    {"ControlledFixtureRoot"},
                )

    def test_symlink_hardlink_and_special_members_are_rejected(self) -> None:
        mutations = ("symlink", "hardlink", "special")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = self._source(base)
                if mutation == "symlink":
                    target = base / "outside.txt"
                    target.write_bytes(b"outside")
                    (root / "README.md").unlink()
                    (root / "README.md").symlink_to(target)
                elif mutation == "hardlink":
                    target = base / "outside.txt"
                    target.write_bytes(b"outside")
                    (root / "README.md").unlink()
                    os.link(target, root / "README.md")
                else:
                    os.mkfifo(root / "pipe")
                with self.assertRaises(builder.PublicZipError):
                    builder.build_public_zip(root, base / "release.zip", SOURCE_DATE_EPOCH)

    def test_member_mutation_during_bounded_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            original_read = builder.os.read
            mutated = False

            def mutating_read(descriptor: int, size: int) -> bytes:
                nonlocal mutated
                payload = original_read(descriptor, size)
                if payload == b"public source fixture\n" and not mutated:
                    mutated = True
                    os.fchmod(descriptor, 0o600)
                return payload

            with mock.patch.object(builder.os, "read", side_effect=mutating_read):
                with self.assertRaisesRegex(builder.PublicZipError, "changed while reading"):
                    builder.build_public_zip(root, base / "release.zip", SOURCE_DATE_EPOCH)
            self.assertTrue(mutated)
            self.assertFalse((base / "release.zip").exists())

    def test_source_mutation_after_capture_rolls_back_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = self._source(base)
            output = base / "release.zip"
            original_write_zip = builder._write_zip

            def mutating_write(*args: object, **kwargs: object) -> None:
                original_write_zip(*args, **kwargs)
                (root / "README.md").chmod(0o600)

            with mock.patch.object(builder, "_write_zip", side_effect=mutating_write):
                with self.assertRaisesRegex(builder.PublicZipError, "changed after snapshot capture"):
                    builder.build_public_zip(root, output, SOURCE_DATE_EPOCH)
            self.assertFalse(output.exists())
            self.assertEqual(list(base.glob(".release.zip.public-zip-*.tmp")), [])

    def test_prohibited_private_components_and_names_are_rejected(self) -> None:
        for relative in ("private_records/note.txt", "reviewer_comments.md"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = self._source(base, extra_paths={relative: b"must not ship\n"})
                with self.assertRaisesRegex(builder.PublicZipError, "prohibited"):
                    builder.build_public_zip(root, base / "release.zip", SOURCE_DATE_EPOCH)


if __name__ == "__main__":
    unittest.main()
