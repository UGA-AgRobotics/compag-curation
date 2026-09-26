"""Focused tests for the separately downloaded public ten-photo release assets."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path


class PrepareTenPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["COMPAG_PUBLIC_ROOT"]).resolve(strict=True)
        cls.model = Path(os.environ["COMPAG_PUBLIC_MODEL_ZIP"]).resolve(strict=True)
        cls.photos = Path(os.environ["COMPAG_PUBLIC_PHOTO_ZIP"]).resolve(strict=True)
        spec = importlib.util.spec_from_file_location("public_example_preparer", cls.root / "tools/prepare_r92_public_example.py")
        assert spec and spec.loader
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def test_all_ten_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "staged"
            result = self.helper.prepare(self.model, self.photos, out)
            self.assertEqual(result["photo_count"], 10)
            self.assertEqual(len(result["cards"]), 10)
            self.assertEqual(len(list((out / "cards").glob("*/IMG_*.jpg"))), 10)
            manifest = json.loads((self.root / "PUBLIC_PHOTO_MANIFEST.json").read_text())
            for row in manifest["photos"]:
                self.assertEqual((out / "cards" / row["source_card_identity"] / row["original_filename"]).read_bytes(),
                                 zipfile.ZipFile(self.photos).read(f"{self.helper.PHOTO_ROOT}/{row['relative_public_path']}"))
            with self.assertRaises(FileExistsError):
                self.helper.prepare(self.model, self.photos, out)

    def _reject_mutated_pack(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.helper.PHOTO_NAME
            with zipfile.ZipFile(self.photos) as source:
                rows = [(info.filename, source.read(info.filename)) for info in source.infolist()]
            rows = mutate(rows)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for name, data in rows:
                    output.writestr(name, data)
            old = self.helper.PHOTO_SHA256
            self.helper.PHOTO_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                with self.assertRaises(ValueError):
                    self.helper.prepare(self.model, path, Path(directory) / "out")
                self.assertFalse((Path(directory) / "out").exists())
            finally:
                self.helper.PHOTO_SHA256 = old

    def test_missing_member_rejected(self) -> None:
        self._reject_mutated_pack(lambda rows: rows[:-1])

    def test_extra_member_rejected(self) -> None:
        self._reject_mutated_pack(lambda rows: rows + [(f"{self.helper.PHOTO_ROOT}/unexpected.txt", b"x")])

    def test_altered_photo_rejected(self) -> None:
        self._reject_mutated_pack(lambda rows: [(n, b"corrupt" if n.endswith("IMG_9317.jpg") else d) for n,d in rows])

    def test_duplicate_member_rejected(self) -> None:
        self._reject_mutated_pack(lambda rows: rows + [rows[-1]])

    def test_unsafe_path_rejected(self) -> None:
        self._reject_mutated_pack(lambda rows: rows[:-1] + [(f"{self.helper.PHOTO_ROOT}/../escape.jpg", rows[-1][1])])

    def test_tampered_model_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / self.helper.MODEL_NAME
            target.write_bytes(self.model.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                self.helper.prepare(target, self.photos, Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
