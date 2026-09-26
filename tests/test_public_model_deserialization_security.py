from __future__ import annotations

import hashlib
import ast
import inspect
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import xgboost as xgb

from compag_curation.proposals.sam2_pipeline import xgb_utils
from compag_curation.proposals.sam2_pipeline import pipeline
from compag_curation.proposals.sam2_pipeline.args import make_argparser


class PublicModelDeserializationSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.safe_json = (
            b'{"learner":{"attributes":{}},"version":[2,1,1]}'
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, name: str, payload: bytes | None = None) -> Path:
        path = self.root / name
        path.write_bytes(self.safe_json if payload is None else payload)
        return path

    def _assert_rejected_without_joblib(self, path: Path, *args: object) -> None:
        with (
            mock.patch.object(
                joblib,
                "load",
                side_effect=AssertionError("joblib.load must not be called"),
            ) as unsafe_joblib_load,
            mock.patch.object(
                pickle,
                "load",
                side_effect=AssertionError("pickle.load must not be called"),
            ) as unsafe_pickle_load,
        ):
            with self.assertRaises((ValueError, RuntimeError)):
                xgb_utils.load_xgb_model(str(path), *args)
        unsafe_joblib_load.assert_not_called()
        unsafe_pickle_load.assert_not_called()

    def test_run_v2_classifier_numeric_private_token_uses_decoded_metadata(self) -> None:
        from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER
        from compag_curation.model_bundle import (
            BUNDLE_SCHEMA_V2,
            _validate_canonical_ubj,
        )
        from compag_curation.public_io import (
            PublicIOError,
            scan_file_for_forbidden_bytes,
            write_new_bytes,
            write_new_json,
        )
        from compag_curation.public_pipeline import (
            _RUN_CLASSIFIER_RELATIVE,
            _RUN_PRIVATE_LOCATOR_TOKENS,
            _validated_run_member_identity,
        )

        matrix = np.zeros(
            (32, len(CANONICAL_FEATURE_ORDER)),
            dtype=np.float32,
        )
        matrix[:, 0] = np.arange(32, dtype=np.float32) % 2
        labels = np.arange(32, dtype=np.float32) % 2
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "max_depth": 1,
                "eta": 1.0,
                "nthread": 1,
                "seed": 42,
            },
            xgb.DMatrix(
                matrix,
                label=labels,
                feature_names=list(CANONICAL_FEATURE_ORDER),
            ),
            num_boost_round=1,
        )
        classifier = bytes(booster.save_raw(raw_format="ubj"))
        adjacent_weights = b"\xbf\xcc\xcc\xcd\x3f\xcc\xcc\xcd"
        self.assertEqual(classifier.count(adjacent_weights), 2)
        offset = classifier.index(adjacent_weights)
        classifier = (
            classifier[:offset]
            + b"/home/\x00\x00"
            + classifier[offset + 8 :]
        )
        self.assertEqual(classifier.count(b"/home/"), 1)

        parsed = xgb.Booster()
        parsed.load_model(bytearray(classifier))
        self.assertEqual(bytes(parsed.save_raw(raw_format="ubj")), classifier)
        _validate_canonical_ubj(classifier)

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "model_bundle"
            bundle.mkdir()
            classifier_path = bundle / "classifier.ubj"
            write_new_json(
                bundle / "bundle.json",
                {"schema": BUNDLE_SCHEMA_V2},
            )
            write_new_bytes(classifier_path, classifier)

            with self.assertRaisesRegex(PublicIOError, "private locator"):
                scan_file_for_forbidden_bytes(
                    classifier_path,
                    _RUN_PRIVATE_LOCATOR_TOKENS,
                )
            digest, snapshot = _validated_run_member_identity(
                classifier_path,
                _RUN_CLASSIFIER_RELATIVE,
            )
            self.assertEqual(
                digest,
                hashlib.sha256(classifier).hexdigest(),
            )
            self.assertEqual(snapshot.st_size, len(classifier))

    def test_dangerous_and_unknown_extensions_are_rejected_before_joblib(self) -> None:
        names = (
            "model.pkl",
            "model.PKL",
            "model.pickle",
            "model.PICKLE",
            "model.joblib",
            "model.JOBLIB",
            "model.bin",
            "model.BIN",
            "model.model",
            "model.txt",
            "model",
        )
        for name in names:
            with self.subTest(name=name):
                self._assert_rejected_without_joblib(self._write(name))

    def test_pickle_payloads_with_safe_suffixes_are_rejected_before_joblib(self) -> None:
        for protocol in (0, pickle.HIGHEST_PROTOCOL):
            payload = pickle.dumps({"payload": "untrusted"}, protocol=protocol)
            for suffix in (".json", ".ubj"):
                with self.subTest(protocol=protocol, suffix=suffix):
                    self._assert_rejected_without_joblib(
                        self._write(f"pickle-{protocol}{suffix}", payload)
                    )

    def test_directories_symbolic_links_and_hard_links_are_rejected(self) -> None:
        directory = self.root / "directory.json"
        directory.mkdir()
        self._assert_rejected_without_joblib(directory)

        target = self._write("target.json")
        symbolic = self.root / "symbolic.json"
        symbolic.symlink_to(target)
        self._assert_rejected_without_joblib(symbolic)

        hard_source = self._write("hard-source.json")
        hard_link = self.root / "hard-link.json"
        os.link(hard_source, hard_link)
        self._assert_rejected_without_joblib(hard_link)

    def test_hash_mismatch_and_invalid_hash_are_rejected_before_model_loading(self) -> None:
        path = self._write("hash.json")
        with mock.patch.object(
            xgb.Booster,
            "load_model",
            side_effect=AssertionError("XGBoost must not receive mismatched bytes"),
        ) as model_load:
            self._assert_rejected_without_joblib(path, "0" * 64)
            self._assert_rejected_without_joblib(path, "A" * 64)
        model_load.assert_not_called()

    def test_path_replacement_during_read_is_rejected(self) -> None:
        path = self._write("stable.json")
        replacement = self._write("replacement.json", self.safe_json + b"\n")
        preserved = self.root / "preserved-original.json"
        real_fstat = os.fstat
        calls = 0

        def swap_on_completed_read(descriptor: int) -> os.stat_result:
            nonlocal calls
            result = real_fstat(descriptor)
            calls += 1
            if calls == 2:
                path.rename(preserved)
                replacement.rename(path)
            return result

        with mock.patch.object(os, "fstat", side_effect=swap_on_completed_read):
            self._assert_rejected_without_joblib(path)
        self.assertEqual(preserved.read_bytes(), self.safe_json)
        self.assertEqual(path.read_bytes(), self.safe_json + b"\n")

    def test_valid_synthetic_json_and_ubj_models_load_from_verified_bytes(self) -> None:
        features = np.arange(16, dtype=np.float32).reshape(8, 2)
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "max_depth": 1,
                "min_child_weight": 0,
                "eta": 1.0,
                "nthread": 1,
                "seed": 42,
            },
            xgb.DMatrix(features, label=labels),
            num_boost_round=2,
        )
        paths = (self.root / "model.json", self.root / "model.ubj")
        for path in paths:
            booster.save_model(path)

        self.assertEqual(xgb.__version__, "2.1.1")
        with (
            mock.patch.object(
                joblib,
                "load",
                side_effect=AssertionError("joblib.load must not be called"),
            ) as unsafe_joblib_load,
            mock.patch.object(
                pickle,
                "load",
                side_effect=AssertionError("pickle.load must not be called"),
            ) as unsafe_pickle_load,
        ):
            for path in paths:
                with self.subTest(suffix=path.suffix):
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    probability = xgb_utils.load_xgb_model(str(path), digest)
                    self.assertIsNotNone(probability)
                    value = probability(features[0])
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
        unsafe_joblib_load.assert_not_called()
        unsafe_pickle_load.assert_not_called()

    def test_public_cli_help_advertises_only_safe_model_formats(self) -> None:
        actions = [
            item for item in make_argparser()._actions if item.dest == "xgb_model"
        ]
        self.assertEqual(len(actions), 2)
        help_text = " ".join(str(action.help).lower() for action in actions)
        self.assertIn(".json", help_text)
        self.assertIn(".ubj", help_text)
        for unsafe in (".pkl", ".pickle", ".joblib", ".bin"):
            self.assertNotIn(unsafe, help_text)

    def test_reachable_pipeline_binds_expected_hash_at_final_load(self) -> None:
        tree = ast.parse(inspect.getsource(pipeline.run_pipeline))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_xgb_model"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 2)
        rendered = ast.unparse(calls[0].args[1])
        self.assertIn("xgb_model_sha256", rendered)


if __name__ == "__main__":
    unittest.main()
