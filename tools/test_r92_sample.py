"""Positive and negative checks for the separately distributed native r92 sample."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from compag_curation.r92_sample import (
    ARCHIVE_SHA256, _payloads, cache_local, download, load_model,
    read_feature_csv, score_csv, synthetic_sample,
)

MODEL = Path(os.environ["COMPAG_TEST_R92_MODEL"])


class R92SampleTests(unittest.TestCase):
    def test_real_model_single_batch_nan_and_synthetic_fixture(self) -> None:
        model = load_model(MODEL)
        base = np.tile(model.medians, (3, 1))
        for name in model.order:
            if name.startswith("g_"):
                base[:, model.order.index(name)] = np.nan
        batch = model.predict(base)
        singles = np.asarray([model.predict(base[i:i + 1])[0] for i in range(3)], dtype=np.float32)
        np.testing.assert_array_equal(batch, singles)
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(batch.shape, (3,))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "sample with spaces"
            receipt = synthetic_sample(MODEL, out)
            self.assertEqual(receipt["row_count"], 3)
            with (out / "scored/detections.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_wrong_hash_missing_member_and_swapped_transform_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wrong = root / "wrong.zip"
            payload = bytearray(MODEL.read_bytes())
            payload[-20] ^= 1
            wrong.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "size/hash"):
                _payloads(wrong)
            with zipfile.ZipFile(MODEL) as archive:
                archive.extractall(root)
            extracted = root / "compag_cj_r92"
            self.assertEqual(len(_payloads(extracted)), 12)
            (extracted / "pca32_mean.npy").write_bytes((extracted / "prototype.npy").read_bytes())
            with self.assertRaisesRegex(ValueError, "identity"):
                _payloads(extracted)
            (extracted / "classifier.ubj").unlink()
            with self.assertRaisesRegex(ValueError, "members"):
                _payloads(extracted)

    def test_schema_invalid_values_cache_and_url_gate(self) -> None:
        model = load_model(MODEL)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "features.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["img_folder", "image", "id", *model.order])
                writer.writeheader()
                writer.writerow({"img_folder":"SYNTHETIC", "image":"SYNTHETIC_y00000x00000.jpg", "id":"1", **{n:"0" for n in model.order}})
            receipt = score_csv(MODEL, source, root / "scored")
            self.assertEqual(receipt["row_count"], 1)
            with self.assertRaises(FileExistsError):
                score_csv(MODEL, source, root / "scored")
            with source.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["img_folder", "image", "id", *model.order])
                writer.writerow({"img_folder":"SYNTHETIC", "image":"SYNTHETIC_y00000x00000.jpg", "id":"1", **{n:"0" for n in model.order}})
            with self.assertRaisesRegex(ValueError, "duplicated"):
                read_feature_csv(source, model.order)
            cached = cache_local(MODEL, root / "cache")
            self.assertEqual(cached.stat().st_size, MODEL.stat().st_size)
            self.assertEqual(ARCHIVE_SHA256, __import__("hashlib").sha256(cached.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                download("http://127.0.0.1/not-a-release.zip", root / "download")


if __name__ == "__main__":
    unittest.main(verbosity=2)
