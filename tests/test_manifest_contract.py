from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import unicodedata
from pathlib import Path


FIELDS = (
    "relative_path",
    "size_bytes",
    "sha256",
    "mode_octal",
    "license_class",
)


def _row(root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    info = path.lstat()
    return {
        "relative_path": relative,
        "size_bytes": str(info.st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode_octal": f"{stat.S_IMODE(info.st_mode):04o}",
        "license_class": "MIT",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]], fields=FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o644)


class ManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public_root = Path(os.environ["COMPAG_PUBLIC_ROOT"]).resolve(strict=True)

    def _tree(self, directory: str) -> tuple[Path, Path]:
        root = Path(directory).resolve() / "release"
        (root / "nested").mkdir(parents=True)
        (root / "payload.txt").write_text("payload\n", encoding="utf-8")
        (root / "nested/config.json").write_text("{}\n", encoding="utf-8")
        for path in (root / "payload.txt", root / "nested/config.json"):
            path.chmod(0o644)
        manifest = root / "manifest.csv"
        _write_manifest(manifest, [_row(root, "nested/config.json"), _row(root, "payload.txt")])
        return root, manifest

    def test_documented_shipped_manifest_command(self) -> None:
        from compag_curation.cli import main

        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
            rc = main(
                [
                    "verify-artifacts",
                    "--root",
                    str(self.public_root),
                    "--manifest",
                    str(self.public_root / "manifests/public_file_inventory.csv"),
                ]
            )
        self.assertEqual(rc, 0)
        self.assertIn('"status":"PASS"', capture.getvalue())

    def test_exact_schema_and_minimal_happy_path(self) -> None:
        from compag_curation.provenance import MANIFEST_FIELDS, verify_manifest
        from tools.build_public_metadata import PUBLIC_MANIFEST_FIELDS

        self.assertEqual(MANIFEST_FIELDS, FIELDS)
        self.assertEqual(PUBLIC_MANIFEST_FIELDS, FIELDS)
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self._tree(directory)
            result = verify_manifest(root, manifest)
            self.assertEqual(result["verified_rows"], 2)
            self.assertEqual(result["manifest_field"], "license_class")

    def test_tamper_missing_extra_duplicate_and_self_exclusion(self) -> None:
        from compag_curation.provenance import ManifestError, verify_manifest

        mutations = ("bytes", "hash", "size", "mode", "missing", "extra", "duplicate", "self")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, manifest = self._tree(directory)
                rows = [_row(root, "nested/config.json"), _row(root, "payload.txt")]
                if mutation == "bytes":
                    (root / "payload.txt").write_text("changed payload\n", encoding="utf-8")
                elif mutation == "hash":
                    rows[1]["sha256"] = "f" * 64
                    _write_manifest(manifest, rows)
                elif mutation == "size":
                    rows[1]["size_bytes"] = str(int(rows[1]["size_bytes"]) + 1)
                    _write_manifest(manifest, rows)
                elif mutation == "mode":
                    (root / "payload.txt").chmod(0o600)
                elif mutation == "missing":
                    (root / "payload.txt").unlink()
                elif mutation == "extra":
                    (root / "extra.txt").write_text("extra\n", encoding="utf-8")
                elif mutation == "duplicate":
                    _write_manifest(manifest, [*rows, rows[1]])
                elif mutation == "self":
                    rows.append(
                        {
                            "relative_path": "manifest.csv",
                            "size_bytes": "0",
                            "sha256": "0" * 64,
                            "mode_octal": "0644",
                            "license_class": "MIT",
                        }
                    )
                    _write_manifest(manifest, rows)
                with self.assertRaises(ManifestError):
                    verify_manifest(root, manifest)

    def test_unsafe_paths_symlinks_and_collisions(self) -> None:
        from compag_curation.provenance import ManifestError, verify_manifest

        unsafe = ("/absolute", "../escape", "nested\\file", "./payload.txt", ".git/secret")
        for relative in unsafe:
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                root, manifest = self._tree(directory)
                rows = [_row(root, "nested/config.json"), _row(root, "payload.txt")]
                rows[1]["relative_path"] = relative
                _write_manifest(manifest, rows)
                with self.assertRaises(ManifestError):
                    verify_manifest(root, manifest)

        for kind in ("file", "directory", "manifest"):
            with self.subTest(symlink=kind), tempfile.TemporaryDirectory() as directory:
                root, manifest = self._tree(directory)
                if kind == "file":
                    (root / "payload.txt").unlink()
                    (root / "payload.txt").symlink_to("nested/config.json")
                elif kind == "directory":
                    (root / "linked").symlink_to("nested", target_is_directory=True)
                else:
                    target = root / "real-manifest.csv"
                    manifest.rename(target)
                    manifest.symlink_to(target.name)
                with self.assertRaises(ManifestError):
                    verify_manifest(root, manifest)

        for names in (("Case.txt", "case.txt"), ("\u00e9.txt", "e\u0301.txt")):
            with self.subTest(collision=names), tempfile.TemporaryDirectory() as directory:
                root, manifest = self._tree(directory)
                for name in names:
                    (root / name).write_text("collision\n", encoding="utf-8")
                    (root / name).chmod(0o644)
                with self.assertRaises(ManifestError):
                    verify_manifest(root, manifest)

    def test_malformed_schema_and_fields_normalize_to_manifest_error(self) -> None:
        from compag_curation.provenance import ManifestError, verify_manifest

        cases = (
            ("wrong-order", ("size_bytes", "relative_path", "sha256", "mode_octal", "license_class"), None),
            ("missing-field", FIELDS[:-1], None),
            ("extra-field", (*FIELDS, "role"), None),
            ("role-schema", ("relative_path", "size_bytes", "sha256", "mode_octal", "role"), None),
            ("empty", FIELDS, []),
            ("bad-size", FIELDS, {"size_bytes": "01"}),
            ("bad-hash", FIELDS, {"sha256": "ABC"}),
            ("bad-mode", FIELDS, {"mode_octal": "644"}),
            ("blank-license", FIELDS, {"license_class": " "}),
        )
        for label, fields, change in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root, manifest = self._tree(directory)
                rows = [_row(root, "nested/config.json"), _row(root, "payload.txt")]
                if change == []:
                    rows = []
                elif change:
                    rows[0].update(change)
                projected = [{field: row.get(field, "value") for field in fields} for row in rows]
                _write_manifest(manifest, projected, fields)
                with self.assertRaises(ManifestError):
                    verify_manifest(root, manifest)

        for missing in ("root", "manifest"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root, manifest = self._tree(directory)
                target_root = root / "absent" if missing == "root" else root
                target_manifest = root / "absent.csv" if missing == "manifest" else manifest
                with self.assertRaises(ManifestError):
                    verify_manifest(target_root, target_manifest)

        with tempfile.TemporaryDirectory() as directory:
            root, _manifest = self._tree(directory)
            outside = root.parent / "outside.csv"
            outside.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
            with self.assertRaises(ManifestError):
                verify_manifest(root, outside)

    def test_generator_rejects_unsafe_and_collision_trees(self) -> None:
        from tools.build_public_metadata import regular_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload").write_text("safe\n", encoding="utf-8")
            self.assertEqual(regular_files(root), [root / "payload"])
            (root / "linked").symlink_to("payload")
            with self.assertRaises(RuntimeError):
                regular_files(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "A").write_text("a", encoding="utf-8")
            (root / "a").write_text("b", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                regular_files(root)

    def test_public_allowlist_is_fail_closed_before_generation(self) -> None:
        from tools.build_public_metadata import (
            ALLOWLIST_ANCHORS,
            ALLOWLIST_PATH,
            ALLOWLIST_SCHEMA_PATH,
            GENERATED_PATHS,
            build,
            load_public_allowlist,
            validate_public_tree,
        )

        base_paths = sorted(ALLOWLIST_ANCHORS | GENERATED_PATHS)

        def populate(parent: Path, *, extra_paths: tuple[str, ...] = ()) -> Path:
            root = parent / "release"
            root.mkdir(mode=0o755)
            root.chmod(0o755)
            paths = sorted({*base_paths, *extra_paths})
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture\n")
                path.chmod(0o644)
            (root / ALLOWLIST_SCHEMA_PATH).write_bytes(
                (self.public_root / ALLOWLIST_SCHEMA_PATH).read_bytes()
            )
            (root / ALLOWLIST_SCHEMA_PATH).chmod(0o644)
            for path in sorted((item for item in root.rglob("*") if item.is_dir())):
                path.chmod(0o755)
            write_allowlist(root, paths)
            return root

        def write_allowlist(root: Path, paths: list[str], **changes: object) -> None:
            payload: dict[str, object] = {
                "schema": "compag-curation-public-release-allowlist/v1",
                "policy": "DENY_BY_DEFAULT_EXACT_REGULAR_FILES",
                "release_version": "1.9.3",
                "path_encoding": "UTF8_NFC_POSIX_RELATIVE_SORTED_LF_TERMINATED",
                "path_count": len(paths),
                "paths_sha256": hashlib.sha256(
                    "".join(f"{item}\n" for item in paths).encode("ascii")
                ).hexdigest(),
                "paths": paths,
            }
            payload.update(changes)
            path = root / ALLOWLIST_PATH
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            path.chmod(0o644)

        with tempfile.TemporaryDirectory() as directory:
            root = populate(Path(directory))
            expected = load_public_allowlist(root)
            self.assertEqual(list(expected), base_paths)
            self.assertEqual(len(validate_public_tree(root, expected)), len(base_paths))

        malformed_cases = ("count", "digest", "order", "duplicate", "collision")
        for case in malformed_cases:
            with self.subTest(allowlist=case), tempfile.TemporaryDirectory() as directory:
                root = populate(Path(directory))
                paths = list(base_paths)
                changes: dict[str, object] = {}
                if case == "count":
                    changes["path_count"] = len(paths) + 1
                elif case == "digest":
                    changes["paths_sha256"] = "f" * 64
                elif case == "order":
                    paths.reverse()
                elif case == "duplicate":
                    paths.append(paths[-1])
                else:
                    paths.extend(("A.txt", "a.txt"))
                    paths.sort()
                write_allowlist(root, paths, **changes)
                with self.assertRaises(RuntimeError):
                    load_public_allowlist(root)

        with tempfile.TemporaryDirectory() as directory:
            root = populate(Path(directory))
            path = root / ALLOWLIST_PATH
            payload = path.read_text(encoding="ascii")
            path.write_text(payload.replace('"schema":', '"schema":"duplicate",\n  "schema":', 1), encoding="ascii")
            path.chmod(0o644)
            with self.assertRaises(RuntimeError):
                load_public_allowlist(root)

        mutations = ("missing", "extra-file", "extra-directory", "symlink", "hardlink", "file-mode", "directory-mode")
        for mutation in mutations:
            with self.subTest(tree=mutation), tempfile.TemporaryDirectory() as directory:
                root = populate(Path(directory))
                expected = load_public_allowlist(root)
                target = root / "manifests/NOTEBOOK_MIGRATION_INVENTORY.csv"
                if mutation == "missing":
                    target.unlink()
                elif mutation == "extra-file":
                    (root / "extra.txt").write_text("extra\n", encoding="ascii")
                elif mutation == "extra-directory":
                    (root / "extra-directory").mkdir(mode=0o755)
                elif mutation == "symlink":
                    target.unlink()
                    target.symlink_to("NOTEBOOK_MIGRATION_INVENTORY.json")
                elif mutation == "hardlink":
                    os.link(target, root / "hardlink.txt")
                elif mutation == "file-mode":
                    target.chmod(0o600)
                else:
                    (root / "manifests").chmod(0o700)
                before = {
                    path.relative_to(root).as_posix(): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink()
                }
                with self.assertRaises(RuntimeError):
                    build(root)
                after = {
                    path.relative_to(root).as_posix(): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink()
                }
                self.assertEqual(after, before)
if __name__ == "__main__":
    unittest.main()
