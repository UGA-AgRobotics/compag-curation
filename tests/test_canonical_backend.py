from __future__ import annotations

import hashlib
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_OPENCV = importlib.util.find_spec("cv2") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


class CanonicalSpecTests(unittest.TestCase):
    def test_exact_93_feature_resource_identity(self) -> None:
        from compag_curation.canonical.spec import (
            CANONICAL_FEATURE_ORDER,
            CANONICAL_FEATURE_ORDER_SHA256,
            canonical_feature_order_bytes,
        )

        root = Path(__file__).resolve().parents[1]
        payload = (
            root
            / "src/compag_curation/resources/canonical_feature_order.txt"
        ).read_bytes()
        self.assertEqual(len(CANONICAL_FEATURE_ORDER), 93)
        self.assertEqual(len(set(CANONICAL_FEATURE_ORDER)), 93)
        self.assertEqual(payload, canonical_feature_order_bytes())
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47",
        )
        self.assertEqual(hashlib.sha256(payload).hexdigest(), CANONICAL_FEATURE_ORDER_SHA256)

    def test_exact_hiera_large_amg_settings_are_immutable(self) -> None:
        from compag_curation.canonical.spec import CanonicalAMGSettings

        settings = CanonicalAMGSettings()
        self.assertEqual(settings.architecture, "sam2.1_hiera_large")
        self.assertEqual(settings.proposal_scales, (1.0,))
        self.assertFalse(settings.yolo_enabled)
        self.assertEqual(
            settings.generator_kwargs(),
            {
                "points_per_side": 64,
                "points_per_batch": 512,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.88,
                "crop_n_layers": 0,
                "crop_n_points_downscale_factor": 2,
                "crop_overlap_ratio": 0.40,
            },
        )
        with self.assertRaises(ValueError):
            CanonicalAMGSettings(points_per_side=32)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required")
class CanonicalTilingTests(unittest.TestCase):
    def test_far_edge_positions_and_edge_padding(self) -> None:
        import numpy as np
        from compag_curation.canonical.preprocessing import (
            canonical_axis_positions,
            iter_canonical_tiles,
        )

        self.assertEqual(canonical_axis_positions(512), (0,))
        self.assertEqual(canonical_axis_positions(513), (0, 1))
        self.assertEqual(canonical_axis_positions(1025), (0, 512, 513))
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        tile = iter_canonical_tiles(image, "card.png")[0]
        self.assertEqual(tile.name, "card_y00000x00000.jpg")
        self.assertEqual(tile.pixels.shape, (512, 512, 3))
        self.assertTrue(np.array_equal(tile.pixels[4, 6], image[4, 6]))
        self.assertTrue(np.array_equal(tile.pixels[-1, -1], image[-1, -1]))

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is required")
    def test_identity_fallback_golden_fixture(self) -> None:
        import numpy as np
        from compag_curation.canonical.preprocessing import canonical_prepare_images

        image = np.zeros((120, 160, 3), dtype=np.uint8)
        prepared = canonical_prepare_images(image, "blank.png", locate_grid=False)
        self.assertEqual(prepared.warp_mode, "identity_fallback")
        self.assertTrue(np.array_equal(prepared.warped_bgr, image))
        self.assertTrue(np.array_equal(prepared.inverse_warp, np.eye(3, dtype=np.float32)))
        self.assertEqual(prepared.evidence["warp"]["fallback"], "identity")

    @unittest.skipUnless(HAS_OPENCV, "OpenCV is required")
    def test_yellow_quadrilateral_golden_fixture(self) -> None:
        import cv2
        import numpy as np
        from compag_curation.canonical.preprocessing import canonical_prepare_images

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        polygon = np.asarray([[60, 30], [270, 45], [290, 210], [35, 195]], np.int32)
        cv2.fillConvexPoly(image, polygon, (0, 255, 255))
        prepared = canonical_prepare_images(image, "quad.png", locate_grid=False)
        self.assertEqual(prepared.warp_mode, "yellow_card_quadrilateral")
        self.assertNotEqual(prepared.warped_bgr.shape, image.shape)
        self.assertEqual(
            prepared.evidence["warp"]["hsv_lower"], [15, 40, 60]
        )
        self.assertEqual(
            prepared.evidence["warp"]["approx_poly_dp_epsilon_perimeter_fraction"],
            0.02,
        )


@unittest.skipUnless(HAS_NUMPY, "NumPy is required")
class CanonicalProposalTests(unittest.TestCase):
    def test_full_masks_use_score_order_and_iou_merge_without_component_reduction(self) -> None:
        import numpy as np
        from compag_curation.canonical.proposals import select_canonical_proposals

        first = np.zeros((12, 12), dtype=np.uint8)
        first[1:3, 1:3] = 1
        first[9:11, 9:11] = 1
        second = np.zeros((12, 12), dtype=np.uint8)
        second[4:7, 5:8] = 1
        result = select_canonical_proposals(
            [
                {"segmentation": first, "predicted_iou": 0.81, "stability_score": 0.89},
                {"segmentation": first.copy(), "predicted_iou": 0.99, "stability_score": 0.99},
                {"segmentation": second, "predicted_iou": 0.82, "stability_score": 0.90},
            ]
        )
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual([item.source_index for item in result.proposals], [1, 2])
        self.assertEqual(result.proposals[0].area, 8)
        self.assertEqual(result.proposals[0].bbox, (1, 1, 10, 10))
        self.assertTrue(np.array_equal(result.proposals[0].mask, first))
        self.assertFalse(result.proposals[0].mask.flags.writeable)

    def test_only_post_dedupe_cap_500_is_applied(self) -> None:
        import numpy as np
        from compag_curation.canonical.proposals import select_canonical_proposals

        annotations = []
        for index in range(501):
            mask = np.zeros((23, 23), dtype=np.uint8)
            mask[index // 23, index % 23] = 1
            annotations.append(
                {"segmentation": mask, "predicted_iou": 0.9, "stability_score": 0.9}
            )
        result = select_canonical_proposals(annotations)
        self.assertEqual(len(result.proposals), 500)
        self.assertEqual(result.capped_count, 1)
        self.assertEqual(result.proposals[-1].source_index, 499)

    def test_archived_scores_are_rounded_only_after_raw_score_selection(self) -> None:
        import numpy as np
        from compag_curation.canonical.proposals import select_canonical_proposals

        mask = np.ones((3, 3), dtype=np.uint8)
        result = select_canonical_proposals(
            [
                {
                    "segmentation": mask,
                    "predicted_iou": 0.90000041,
                    "stability_score": 0.81234561,
                },
                {
                    "segmentation": mask.copy(),
                    "predicted_iou": 0.90000044,
                    "stability_score": 0.81234569,
                },
            ]
        )

        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.proposals[0].source_index, 1)
        self.assertEqual(result.proposals[0].predicted_iou, 0.9)
        self.assertEqual(result.proposals[0].stability_score, 0.812346)

    def test_raw_stability_breaks_iou_ties_before_archival_rounding(self) -> None:
        import numpy as np
        from compag_curation.canonical.proposals import select_canonical_proposals

        mask = np.ones((2, 2), dtype=np.uint8)
        result = select_canonical_proposals(
            [
                {
                    "segmentation": mask,
                    "predicted_iou": 0.9,
                    "stability_score": 0.70000041,
                },
                {
                    "segmentation": mask.copy(),
                    "predicted_iou": 0.9,
                    "stability_score": 0.70000044,
                },
            ]
        )

        self.assertEqual(result.proposals[0].source_index, 1)
        self.assertEqual(result.proposals[0].stability_score, 0.7)

    def test_archival_rounding_does_not_launder_invalid_source_scores(self) -> None:
        import numpy as np
        from compag_curation.canonical.proposals import select_canonical_proposals

        mask = np.ones((2, 2), dtype=np.uint8)
        for value in (-0.0000001, 1.0000001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_canonical_proposals(
                    [
                        {
                            "segmentation": mask,
                            "predicted_iou": value,
                            "stability_score": 0.9,
                        }
                    ]
                )


@unittest.skipUnless(HAS_NUMPY and HAS_OPENCV, "NumPy and OpenCV are required")
class CanonicalFeatureTests(unittest.TestCase):
    def _raw_fixture(self):
        import numpy as np
        from compag_curation.canonical.features import (
            CanonicalGridContext,
            extract_canonical_raw_features,
        )
        from compag_curation.canonical.proposals import select_canonical_proposals

        image = np.empty((64, 64, 3), dtype=np.uint8)
        image[:] = (20, 180, 220)
        image[20:35, 20:35] = (30, 30, 30)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:35, 20:35] = 1
        proposals = select_canonical_proposals(
            [{"segmentation": mask, "predicted_iou": 0.9, "stability_score": 0.95}]
        ).proposals

        class StaticEmbedder:
            def embed_many(self, patches):
                vectors = np.zeros((len(patches), 2048), dtype=np.float32)
                for index in range(len(patches)):
                    vectors[index, index] = 1.0
                return vectors

        return extract_canonical_raw_features(
            image,
            proposals,
            StaticEmbedder(),
            grid=CanonicalGridContext((), ()),
        )

    def _feature_state(self):
        import numpy as np
        from compag_curation.canonical.features import CanonicalFeatureState

        prototype = np.zeros(2048, dtype=np.float32)
        prototype[0] = 1.0
        components = np.zeros((32, 2048), dtype=np.float32)
        components[np.arange(32), np.arange(32)] = 1.0
        return CanonicalFeatureState(
            prototype=prototype,
            pca_components=components,
            pca_mean=np.zeros(2048, dtype=np.float32),
            training_row_count=32,
            positive_row_count=16,
        )

    def test_resnet_weights_swap_restore_reads_verified_descriptor_or_fails_closed(self) -> None:
        from unittest import mock

        from compag_curation.canonical import features
        from compag_curation.canonical.features import CanonicalResNet50Embedder
        from compag_curation.public_io import PublicIOError

        original = b"verified-resnet-state\n"
        substituted = b"substituted-resnet-state\n"
        consumed = []

        class Model:
            def to(self, _device):
                return self

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "resnet50.pth"
            moved = root / "resnet50.original.pth"
            weights.write_bytes(original)
            expected_sha256 = hashlib.sha256(original).hexdigest()

            def adversarial_load(_backbone, weights_file):
                self.assertTrue(hasattr(weights_file, "fileno"))
                weights.rename(moved)
                weights.write_bytes(substituted)
                try:
                    weights_file.seek(0)
                    consumed.append(weights_file.read())
                finally:
                    weights.unlink()
                    moved.rename(weights)
                return Model(), lambda value: value

            with (
                mock.patch(
                    "compag_curation.proposals.sam2_pipeline.gate_core.build_embed_model",
                    side_effect=adversarial_load,
                ),
                mock.patch.object(features, "_attest_cuda_module"),
            ):
                try:
                    result = CanonicalResNet50Embedder.from_local_weights(
                        weights, expected_sha256
                    )
                except PublicIOError as exc:
                    self.assertIn("verified file", str(exc))
                else:
                    self.assertIsInstance(result, CanonicalResNet50Embedder)

        self.assertEqual(consumed, [original])

    def test_four_export_scales_and_complete_exact_order_rows(self) -> None:
        import numpy as np
        from compag_curation.canonical.features import finalize_canonical_feature
        from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER

        raw = self._raw_fixture()
        self.assertEqual([item.scale for item in raw], [0.67, 0.80, 1.00, 1.25])
        row = finalize_canonical_feature(raw[0], self._feature_state())
        self.assertEqual(tuple(row), CANONICAL_FEATURE_ORDER)
        self.assertEqual(len(row), 93)
        self.assertTrue(np.isfinite(list(row.values())).all())

    def test_missing_raw_feature_is_rejected_not_zero_filled(self) -> None:
        from compag_curation.canonical.features import (
            CanonicalRawFeature,
            finalize_canonical_feature,
        )

        source = self._raw_fixture()[0]
        incomplete = dict(source.values)
        del incomplete["std_b"]
        raw = CanonicalRawFeature(
            proposal_index=source.proposal_index,
            scale=source.scale,
            predicted_iou=source.predicted_iou,
            stability_score=source.stability_score,
            values=incomplete,
            embedding=source.embedding,
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            finalize_canonical_feature(raw, self._feature_state())

    def test_raw_csv_embedding_codec_is_lossless_float32(self) -> None:
        import numpy as np
        from compag_curation.canonical.features import (
            CANONICAL_RAW_FEATURE_COLUMNS,
            canonical_raw_feature_csv_row,
            decode_canonical_embedding,
        )

        raw = self._raw_fixture()[0]
        row = canonical_raw_feature_csv_row(raw)
        decoded = decode_canonical_embedding(row["embedding_f32le_base64"])
        self.assertEqual(tuple(row), CANONICAL_RAW_FEATURE_COLUMNS)
        self.assertEqual(row["embedding_encoding"], "base64-float32-little-endian-v1")
        self.assertEqual(row["embedding_dimensions"], 2048)
        self.assertEqual(decoded.tobytes(), np.asarray(raw.embedding, dtype="<f4").tobytes())
        with self.assertRaises(ValueError):
            decode_canonical_embedding(row["embedding_f32le_base64"][:-4])

    def test_feature_state_is_fitted_only_on_declared_train_rows(self) -> None:
        import numpy as np
        from compag_curation.canonical.features import (
            CanonicalRawFeature,
            fit_canonical_feature_state,
        )

        raw = self._raw_fixture()[0]
        rows = []
        labels = []
        for index in range(36):
            vector = np.zeros(2048, dtype=np.float32)
            vector[index if index < 32 else 2000 + index] = 1.0
            rows.append(
                CanonicalRawFeature(index, 1.0, 0.9, 0.9, raw.values, vector)
            )
            labels.append(index % 2)
        state = fit_canonical_feature_state(rows, labels, range(32))
        expected_mean = np.zeros(2048, dtype=np.float32)
        expected_mean[:32] = 1.0 / 32.0
        self.assertTrue(np.allclose(state.pca_mean, expected_mean, atol=1e-7))
        self.assertEqual(state.pca_components.shape, (32, 2048))
        self.assertEqual(state.positive_row_count, 16)


@unittest.skipUnless(HAS_NUMPY and HAS_OPENCV and HAS_TORCH, "Torch stack is required")
class CanonicalBatchEmbeddingTests(unittest.TestCase):
    def test_embed_model_accepts_legacy_paths_and_caller_owned_descriptors(self) -> None:
        from unittest import mock

        import torch
        from torchvision import models

        from compag_curation.proposals.sam2_pipeline.gate_core import (
            build_embed_model,
        )

        payload = b"fixture-resnet-state\n"
        consumed = []
        loaded_states = []

        class Model:
            def load_state_dict(self, state, *, strict):
                loaded_states.append((state, strict))

            def eval(self):
                return self

        class UntrustedWeightsWrapper:
            def __init__(self, weights_file):
                self._weights_file = weights_file

            def fileno(self):
                return self._weights_file.fileno()

            def read(self, *_args, **_kwargs):
                raise AssertionError("the caller-owned read method must not be used")

            def seek(self, *_args, **_kwargs):
                raise AssertionError("the caller-owned seek method must not be used")

        def fake_load(weights_file, *, map_location, weights_only):
            self.assertEqual(map_location, "cpu")
            self.assertTrue(weights_only)
            self.assertTrue(hasattr(weights_file, "fileno"))
            consumed.append(weights_file.read())
            return {"fixture": len(consumed)}

        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "resnet50.pth"
            weights.write_bytes(payload)
            with (
                mock.patch.object(models, "resnet50", side_effect=lambda **_kwargs: Model()),
                mock.patch.object(torch, "load", side_effect=fake_load),
            ):
                path_model, _path_transform = build_embed_model("resnet50", weights)
                with weights.open("rb") as weights_file:
                    descriptor_model, _descriptor_transform = build_embed_model(
                        "resnet50", UntrustedWeightsWrapper(weights_file)
                    )
                    self.assertFalse(weights_file.closed)

        self.assertIsInstance(path_model.fc, torch.nn.Identity)
        self.assertIsInstance(descriptor_model.fc, torch.nn.Identity)
        self.assertEqual(consumed, [payload, payload])
        self.assertEqual(
            loaded_states,
            [({"fixture": 1}, True), ({"fixture": 2}, True)],
        )

    def test_batched_and_scalar_masked_crop_embeddings_match(self) -> None:
        import numpy as np
        import torch
        from compag_curation.canonical.features import CanonicalResNet50Embedder

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, values):
                means = values.mean(dim=(1, 2, 3)).reshape(-1, 1)
                offsets = torch.arange(
                    2048, dtype=values.dtype, device=values.device
                ).reshape(1, -1) / 2048.0
                return means + offsets

        def transform(rgb):
            return (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        generator = np.random.default_rng(12)
        patches = [
            generator.integers(0, 256, (18, 21, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        embedder = CanonicalResNet50Embedder(Model(), transform)
        batched = embedder.embed_many(patches)
        scalar = np.vstack([embedder.embed_one_scalar(patch) for patch in patches])
        self.assertTrue(np.allclose(batched, scalar, rtol=0.0, atol=1e-7))


@unittest.skipUnless(HAS_NUMPY, "NumPy is required")
class CanonicalTrainingContractTests(unittest.TestCase):
    def test_cuda_memory_release_is_ordered_and_fail_closed(self) -> None:
        import types
        from unittest import mock

        from compag_curation.canonical import training

        events: list[str] = []

        class Pool:
            def __init__(self, name: str) -> None:
                self.name = name

            def free_all_blocks(self) -> None:
                events.append(self.name)

        runtime = types.SimpleNamespace(
            getDeviceCount=lambda: 1,
            getDevice=lambda: 0,
            deviceSynchronize=lambda: events.append("synchronize"),
        )
        cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(runtime=runtime),
            get_default_memory_pool=lambda: Pool("default-pool"),
            get_default_pinned_memory_pool=lambda: Pool("pinned-pool"),
        )
        with mock.patch.object(
            training.gc,
            "collect",
            side_effect=lambda: events.append("gc") or 0,
        ):
            training._release_cuda_execution_memory("cuda", cupy)

        self.assertEqual(
            events,
            ["gc", "synchronize", "default-pool", "pinned-pool"],
        )
        with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
            training._release_cuda_execution_memory("cpu", cupy)
        broken = types.SimpleNamespace(
            cuda=types.SimpleNamespace(runtime=runtime),
            get_default_memory_pool=lambda: Pool("unused"),
        )
        with self.assertRaisesRegex(RuntimeError, "interface is unavailable"):
            training._release_cuda_execution_memory("cuda", broken)

    def test_fit_releases_early_model_and_final_cuda_inputs_in_order(self) -> None:
        import json
        import sys
        import types
        from unittest import mock

        import numpy as np
        from compag_curation.canonical import training

        events: list[str] = []
        array_number = 0

        class CudaArray:
            def __init__(self, value, number: int) -> None:
                self.value = np.asarray(value)
                self.number = number

            def __len__(self) -> int:
                return len(self.value)

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": self.value.shape,
                    "typestr": self.value.dtype.str,
                    "data": (1, False),
                    "version": 3,
                }

            def __del__(self) -> None:
                events.append(f"array-del:{self.number}")

        class Pool:
            def __init__(self, name: str) -> None:
                self.name = name

            def free_all_blocks(self) -> None:
                events.append(self.name)

        runtime = types.SimpleNamespace(
            getDeviceCount=lambda: 1,
            getDevice=lambda: 0,
            deviceSynchronize=lambda: events.append("synchronize"),
        )
        cupy = types.ModuleType("cupy")
        cupy.cuda = types.SimpleNamespace(runtime=runtime)

        def asarray(value):
            nonlocal array_number
            result = CudaArray(value, array_number)
            array_number += 1
            return result

        cupy.asarray = asarray
        cupy.get_default_memory_pool = lambda: Pool("default-pool")
        cupy.get_default_pinned_memory_pool = lambda: Pool("pinned-pool")

        class Booster:
            def __init__(self, role: str) -> None:
                self.role = role

            def save_config(self) -> str:
                return json.dumps(
                    {
                        "learner": {
                            "generic_param": {
                                "device": "cuda:0",
                                "nthread": "1",
                                "fail_on_invalid_gpu_id": "1",
                            }
                        }
                    }
                )

            def __del__(self) -> None:
                events.append(f"booster-del:{self.role}")

        class Classifier:
            def __init__(self, **parameters) -> None:
                self.role = (
                    "early"
                    if parameters["early_stopping_rounds"] == 30
                    else "final"
                )
                self.best_iteration = 4
                self.booster = Booster(self.role)
                events.append(f"factory:{self.role}")

            def fit(self, *_args, **_kwargs):
                events.append(f"fit:{self.role}")
                return self

            def get_booster(self):
                return self.booster

            def __del__(self) -> None:
                events.append(f"model-del:{self.role}")

        xgboost = types.ModuleType("xgboost")
        xgboost.build_info = lambda: {"USE_CUDA": True}
        xgboost.XGBClassifier = Classifier
        sampled = {
            "smote__sampling_strategy": 0.5,
            "smote__k_neighbors": 3,
        }
        features = np.asarray(
            [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0], [1.0, 4.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 0, 1], dtype=int)
        groups = np.asarray(["a", "b", "c", "d"])
        weights = np.ones(4, dtype=np.float32)

        with (
            mock.patch.dict(
                sys.modules,
                {"xgboost": xgboost, "cupy": cupy},
            ),
            mock.patch.object(
                training,
                "_group_safe_inner_split",
                return_value=(np.asarray([0, 1]), np.asarray([2, 3])),
            ),
            mock.patch.object(
                training.gc,
                "collect",
                side_effect=lambda: events.append("gc") or 0,
            ),
        ):
            model, _imputer, used_trees = training._fit_early_stop_then_full(
                features,
                labels,
                groups,
                weights,
                sampled,
                config=training.CanonicalTrainingConfig(),
                resample=lambda x, y, w, *_args: (x, y, w),
            )
        events.append("returned")

        self.assertEqual(used_trees, 5)
        self.assertEqual(model.role, "final")
        self.assertNotIn("model-del:final", events)
        self.assertEqual(events.count("gc"), 2)
        self.assertEqual(events.count("synchronize"), 2)
        self.assertEqual(events.count("default-pool"), 2)
        self.assertEqual(events.count("pinned-pool"), 2)
        first_gc = events.index("gc")
        second_gc = events.index("gc", first_gc + 1)
        self.assertLess(events.index("model-del:early"), first_gc)
        self.assertLess(events.index("booster-del:early"), first_gc)
        for number in range(5):
            self.assertLess(events.index(f"array-del:{number}"), first_gc)
        self.assertLess(first_gc, events.index("factory:final"))
        for number in range(5, 8):
            self.assertLess(events.index(f"array-del:{number}"), second_gc)
        self.assertLess(second_gc, events.index("returned"))

    def test_weighted_search_releases_every_fold_model_after_scoring(self) -> None:
        import types
        from unittest import mock

        import numpy as np
        from compag_curation.canonical import training

        events: list[str] = []
        model_number = 0

        class Booster:
            def __init__(self, number: int) -> None:
                self.number = number

            def __del__(self) -> None:
                events.append(f"booster-del:{self.number}")

        class Model:
            def __init__(self, number: int) -> None:
                self.number = number
                self.booster = Booster(number)

            def get_booster(self):
                return self.booster

            def __del__(self) -> None:
                events.append(f"model-del:{self.number}")

        class Imputer:
            def transform(self, values):
                return np.asarray(values, dtype=np.float32)

        def fake_fit(*_args, **_kwargs):
            nonlocal model_number
            number = model_number
            model_number += 1
            events.append(f"fit:{number}")
            return Model(number), Imputer(), 5

        def fake_predict(model, _features, *, device):
            self.assertEqual(device, "cuda")
            events.append(f"predict:{model.number}")
            return np.asarray([0.2, 0.8], dtype=np.float32)

        def fake_score(_labels, _probabilities):
            events.append(f"score:{len([e for e in events if e.startswith('score:')])}")
            return 0.75

        features = np.asarray(
            [[0.0], [1.0], [0.0], [1.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 0, 1], dtype=int)
        groups = np.asarray(["a", "b", "c", "d"])
        weights = np.ones(4, dtype=np.float32)
        folds = (
            (np.asarray([2, 3]), np.asarray([0, 1])),
            (np.asarray([0, 1]), np.asarray([2, 3])),
        )
        config = types.SimpleNamespace(
            search_iterations=1,
            group_folds=2,
            device="cuda",
        )
        with (
            mock.patch.object(
                training,
                "_fit_early_stop_then_full",
                new=fake_fit,
            ),
            mock.patch.object(
                training,
                "_positive_probabilities",
                new=fake_predict,
            ),
            mock.patch(
                "sklearn.metrics.average_precision_score",
                side_effect=fake_score,
            ),
            mock.patch.object(
                training,
                "_release_cuda_execution_memory",
                new=lambda *_args: events.append("release"),
            ),
        ):
            result = training._weighted_parameter_search(
                features,
                labels,
                groups,
                weights,
                folds,
                ({"fixture": True},),
                config=config,
            )

        self.assertEqual(result.fold_count, 2)
        self.assertEqual(events.count("release"), 2)
        for number in range(2):
            score = events.index(f"score:{number}")
            model_delete = events.index(f"model-del:{number}")
            booster_delete = events.index(f"booster-del:{number}")
            release = events.index("release", score)
            self.assertLess(score, model_delete)
            self.assertLess(model_delete, release)
            self.assertLess(booster_delete, release)

    def test_supplied_review_weights_are_validated_and_skip_is_zero(self) -> None:
        import numpy as np
        from compag_curation.canonical.training import validate_canonical_review_weights

        actions = ["accept", "flip", "sus_accept", "sus_flip", "skip"]
        weights = validate_canonical_review_weights(actions, [1, 1.0, 0.4, 0.4, 0])
        self.assertTrue(np.array_equal(weights, np.asarray([1, 1, 0.4, 0.4, 0], np.float32)))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_canonical_review_weights(["sus_accept"], [1.0])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_canonical_review_weights([""], [1.0])

    def test_tile_balanced_split_is_deterministic_and_group_pure(self) -> None:
        from compag_curation.canonical.training import canonical_tile_balanced_split

        groups = [name for name, count in (("a", 11), ("b", 7), ("c", 6), ("d", 5), ("e", 4), ("f", 3)) for _ in range(count)]
        first = canonical_tile_balanced_split(groups)
        second = canonical_tile_balanced_split(groups)
        self.assertEqual(first, second)
        self.assertFalse(first.train_groups & first.test_groups)
        self.assertEqual(first.train_groups | first.test_groups, set(groups))
        self.assertEqual(first.target_test_rows, round(len(groups) * 0.2))

    def test_eight_group_demo_fixture_leaves_five_fold_training_capacity(self) -> None:
        from compag_curation.canonical.service import canonical_review_split_feasibility

        rows = [
            {
                "group_id": f"synthetic-group-{group_index:02d}",
                "label": int(proposal_index == 0),
                "review_action": "accept",
                "review_weight": 1.0,
            }
            for group_index in range(1, 9)
            for proposal_index in range(2)
        ]
        result = canonical_review_split_feasibility(rows)
        train_groups = set(result["train_groups"])
        test_groups = set(result["test_groups"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(train_groups | test_groups), 8)
        self.assertFalse(train_groups & test_groups)
        self.assertGreaterEqual(len(train_groups), 5)
        self.assertEqual(result["group_cv_folds"], 5)
        self.assertGreaterEqual(4 * int(result["coverage"]["train"]["rows"]), 32)
        for partition in ("train", "test"):
            self.assertEqual(
                set(result["coverage"][partition]) - {"rows"},
                {"negative", "positive"},
            )
            self.assertGreater(int(result["coverage"][partition]["negative"]), 0)
            self.assertGreater(int(result["coverage"][partition]["positive"]), 0)

    def test_exact_random_parameter_sampler_has_30_rows(self) -> None:
        from compag_curation.canonical.training import canonical_parameter_samples

        first = canonical_parameter_samples()
        second = canonical_parameter_samples()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        for row in first:
            self.assertTrue(3 <= row["clf__max_depth"] <= 9)
            self.assertIn(row["smote__sampling_strategy"], (0.4, 0.5, 0.6))
            self.assertTrue(2 <= row["smote__k_neighbors"] <= 5)
            self.assertEqual(row["clf__max_bin"], 256)

    def test_weighted_search_uses_five_folds_30_settings_and_early_stop_30(self) -> None:
        import numpy as np
        from types import SimpleNamespace
        from compag_curation.canonical.training import (
            CanonicalTrainingConfig,
            _weighted_parameter_search,
        )

        labels = np.asarray([0, 1] * 10, dtype=int)
        features = np.column_stack((labels, np.arange(20), np.ones(20))).astype(np.float32)
        groups = np.asarray([f"g{index // 2}" for index in range(20)])
        weights = np.ones(20, dtype=np.float32)
        folds = []
        all_indices = np.arange(20)
        for fold in range(5):
            validation = np.asarray([fold * 2, fold * 2 + 1])
            train = np.setdiff1d(all_indices, validation)
            folds.append((train, validation))
        samples = [
            {
                "clf__max_depth": 3 + index % 7,
                "clf__learning_rate": 0.05,
                "smote__sampling_strategy": 0.5,
                "smote__k_neighbors": 3,
            }
            for index in range(30)
        ]
        observed_parameters = []

        class Classifier:
            def __init__(self, parameters):
                self.parameters = parameters
                self.fit_values = None
                self.eval_values = None
                self.predicted_values = []

            def fit(self, features, labels, **kwargs):
                self.fit_values = tuple(int(value) for value in features[:, 1])
                evaluation = kwargs.get("eval_set", ())
                self.eval_values = (
                    tuple(int(value) for value in evaluation[0][0][:, 1])
                    if evaluation
                    else None
                )
                return self

            def predict_proba(self, values):
                self.predicted_values.append(tuple(int(value) for value in values[:, 1]))
                positive = np.where(values[:, 0] > 0.5, 0.8, 0.2).astype(np.float32)
                return np.column_stack((1.0 - positive, positive))

        def factory(parameters):
            observed_parameters.append(dict(parameters))
            return Classifier(parameters)

        def identity_resample(features, labels, weights, strategy, neighbors, seed):
            return features, labels, weights

        result = _weighted_parameter_search(
            features,
            labels,
            groups,
            weights,
            folds,
            samples,
            config=SimpleNamespace(
                **{
                    **CanonicalTrainingConfig().__dict__,
                    "device": "cpu",
                }
            ),
            classifier_factory=factory,
            resample=identity_resample,
        )
        self.assertEqual(result.sampled_configuration_count, 30)
        self.assertEqual(result.fold_count, 5)
        self.assertEqual(len(observed_parameters), 300)
        self.assertTrue(
            all(row["n_estimators"] == 2000 for row in observed_parameters)
        )
        self.assertEqual(
            [row["early_stopping_rounds"] for row in observed_parameters].count(30),
            150,
        )
        self.assertEqual(
            [row["early_stopping_rounds"] for row in observed_parameters].count(None),
            150,
        )
        self.assertTrue(all(row["scale_pos_weight"] == 1.0 for row in observed_parameters))
        for pair_index in range(150):
            early = observed_parameters[pair_index * 2]
            final = observed_parameters[pair_index * 2 + 1]
            self.assertEqual(early["early_stopping_rounds"], 30)
            self.assertIsNone(final["early_stopping_rounds"])

    def test_training_uses_all_export_scales_only_for_fit_and_scale_one_for_scoring(self) -> None:
        import numpy as np
        from types import SimpleNamespace
        from unittest import mock
        from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER
        from compag_curation.canonical.training import (
            CanonicalGroupSplit,
            CanonicalSearchResult,
            train_canonical_xgb,
        )

        scales = (0.67, 0.80, 1.00, 1.25)
        rows = []
        labels = []
        groups = []
        observed_masks = []
        for group_index in range(6):
            for label in (0, 1):
                for scale in scales:
                    rows.append({name: float(label + scale) for name in CANONICAL_FEATURE_ORDER})
                    labels.append(label)
                    groups.append(f"g{group_index}")
        supplied_scales = list(scales) * 12
        actions = ["accept"] * len(rows)
        weights = [1.0] * len(rows)
        split = CanonicalGroupSplit(
            train_groups=frozenset(f"g{index}" for index in range(5)),
            test_groups=frozenset({"g5"}),
            target_test_rows=8,
            observed_test_rows=8,
        )

        def fake_search(features, labels, groups, weights, cv_pairs, samples, **kwargs):
            observed_masks.append((len(features), tuple(kwargs["canonical_validation_mask"])))
            return CanonicalSearchResult({}, 0.5, 30, 5, (0.5,) * 30)

        class Imputer:
            statistics_ = np.zeros(len(CANONICAL_FEATURE_ORDER), dtype=np.float32)

            def transform(self, values):
                return np.asarray(values, dtype=np.float32)

        class Model:
            def predict_proba(self, values):
                positive = np.full(len(values), 0.75, dtype=np.float32)
                return np.column_stack((1.0 - positive, positive))

        def fake_fit(features, labels, groups, weights, sampled, **kwargs):
            observed_masks.append((len(features), tuple(kwargs["canonical_validation_mask"])))
            return Model(), Imputer(), 17

        with (
            mock.patch("compag_curation.canonical.training._weighted_parameter_search", side_effect=fake_search),
            mock.patch("compag_curation.canonical.training._fit_early_stop_then_full", side_effect=fake_fit),
            mock.patch(
                "compag_curation.canonical.training._positive_probabilities",
                return_value=np.full(2, 0.75, dtype=np.float32),
            ),
        ):
            result = train_canonical_xgb(
                rows,
                labels,
                groups,
                actions,
                weights,
                scales=supplied_scales,
                split=split,
            )
        self.assertEqual(len(result.test_probabilities), 2)
        self.assertEqual(result.test_groups, ("g5", "g5"))
        self.assertEqual([count for count, _mask in observed_masks], [40, 40])
        self.assertTrue(all(sum(mask) == 10 for _count, mask in observed_masks))


class CanonicalModelLoadingTests(unittest.TestCase):
    def test_loader_rejects_non_hiera_locator_before_importing_sam2(self) -> None:
        from compag_curation.canonical.model_loading import load_canonical_sam2_model
        from compag_curation.canonical.spec import (
            CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
            CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
        )

        with self.assertRaisesRegex(ValueError, "not Hiera-L"):
            load_canonical_sam2_model(
                Path("/abs/config.yaml"),
                Path("/abs/checkpoint.pt"),
                "configs/sam2.1/sam2.1_hiera_t",
                CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
                CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
            )

    def test_loader_registers_descriptor_pinned_config_before_upstream_build(self) -> None:
        import hashlib
        import os
        import sys
        import types
        from unittest import mock

        from compag_curation.canonical import model_loading
        from compag_curation.public_io import PublicIOError

        config_bytes = b"model:\n  sentinel: verified\n"
        checkpoint_bytes = b"verified-checkpoint"
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        registered: dict[str, object] = {}

        class FakeOmegaConf:
            @staticmethod
            def load(path: str) -> dict[str, object]:
                self.assertTrue(path.startswith("/proc/self/fd/"))
                registered["config_bytes"] = Path(path).read_bytes()
                return {"model": {"sentinel": "verified"}}

        class FakeConfigStore:
            @classmethod
            def instance(cls):
                return cls()

            def store(self, **kwargs) -> None:
                registered["store"] = kwargs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root = root / "sam2"
            installed = package_root / "configs/sam2.1/sam2.1_hiera_l.yaml"
            installed.parent.mkdir(parents=True)
            installed.write_bytes(config_bytes)
            package_init = package_root / "__init__.py"
            package_init.write_bytes(b"")
            supplied = root / "registered.yaml"
            supplied.write_bytes(config_bytes)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(checkpoint_bytes)

            fake_sam2 = types.ModuleType("sam2")
            fake_sam2.__file__ = str(package_init)
            fake_build_module = types.ModuleType("sam2.build_sam")

            def fake_build(config_name, checkpoint_path, **kwargs):
                self.assertEqual(
                    config_name, f"compag_verified_sam2_hiera_l_{config_hash}"
                )
                self.assertTrue(str(checkpoint_path).startswith("/proc/self/fd/"))
                self.assertEqual(Path(checkpoint_path).read_bytes(), checkpoint_bytes)
                original = installed.with_name("original.yaml")
                os.replace(installed, original)
                installed.write_bytes(b"model:\n  sentinel: unverified\n")
                try:
                    self.assertEqual(registered["config_bytes"], config_bytes)
                    self.assertEqual(
                        registered["store"]["node"],
                        {"model": {"sentinel": "verified"}},
                    )
                finally:
                    installed.unlink()
                    os.replace(original, installed)
                return "verified-model"

            fake_build_module.build_sam2 = fake_build
            fake_hydra = types.ModuleType("hydra")
            fake_hydra_core = types.ModuleType("hydra.core")
            fake_config_store = types.ModuleType("hydra.core.config_store")
            fake_config_store.ConfigStore = FakeConfigStore
            fake_omegaconf = types.ModuleType("omegaconf")
            fake_omegaconf.OmegaConf = FakeOmegaConf

            with (
                mock.patch.object(
                    model_loading,
                    "CANONICAL_SAM2_HIERA_L_CONFIG_SHA256",
                    config_hash,
                ),
                mock.patch.object(
                    model_loading,
                    "CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256",
                    checkpoint_hash,
                ),
                mock.patch.dict(
                    sys.modules,
                    {
                        "sam2": fake_sam2,
                        "sam2.build_sam": fake_build_module,
                        "hydra": fake_hydra,
                        "hydra.core": fake_hydra_core,
                        "hydra.core.config_store": fake_config_store,
                        "omegaconf": fake_omegaconf,
                    },
                ),
                mock.patch.object(
                    model_loading,
                    "_attest_canonical_sam2_cuda_model",
                    return_value=0,
                ),
            ):
                try:
                    model = model_loading.load_canonical_sam2_model(
                        supplied,
                        checkpoint,
                        "configs/sam2.1/sam2.1_hiera_l",
                        config_hash,
                        checkpoint_hash,
                    )
                except PublicIOError as exc:
                    self.assertIn("verified file", str(exc))
                    model = None
            self.assertEqual(registered["config_bytes"], config_bytes)
            self.assertEqual(installed.read_bytes(), config_bytes)
            if model is not None:
                self.assertEqual(model, "verified-model")


@unittest.skipUnless(HAS_NUMPY, "NumPy is required")
class CanonicalSerializationTests(unittest.TestCase):
    def _state(self):
        import numpy as np
        from compag_curation.canonical.features import CanonicalFeatureState

        prototype = np.zeros(2048, dtype=np.float32)
        prototype[0] = 1.0
        components = np.zeros((32, 2048), dtype=np.float32)
        components[np.arange(32), np.arange(32)] = 1.0
        return CanonicalFeatureState(
            prototype,
            components,
            np.zeros(2048, dtype=np.float32),
            32,
            16,
        )

    def test_feature_state_npy_roundtrip_rejects_trailing_bytes(self) -> None:
        import numpy as np
        from compag_curation.canonical.serialization import (
            CanonicalFeatureStatePayloads,
            deserialize_canonical_feature_state,
            serialize_canonical_feature_state,
        )

        state = self._state()
        payloads = serialize_canonical_feature_state(state)
        loaded = deserialize_canonical_feature_state(
            payloads, training_row_count=32, positive_row_count=16
        )
        self.assertTrue(np.array_equal(loaded.prototype, state.prototype))
        self.assertTrue(np.array_equal(loaded.pca_components, state.pca_components))
        with self.assertRaisesRegex(ValueError, "trailing"):
            deserialize_canonical_feature_state(
                CanonicalFeatureStatePayloads(
                    payloads.prototype_npy + b"x",
                    payloads.pca_components_npy,
                    payloads.pca_mean_npy,
                ),
                training_row_count=32,
                positive_row_count=16,
            )

    @unittest.skipUnless(HAS_XGBOOST, "XGBoost is required")
    def test_ubj_probability_parity_on_synthetic_rows(self) -> None:
        import numpy as np
        from xgboost import XGBClassifier
        from compag_curation.canonical.serialization import verify_probability_parity
        from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER

        generator = np.random.default_rng(7)
        features = generator.normal(size=(40, 93)).astype(np.float32)
        labels = np.arange(40) % 2
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device="cpu",
            n_estimators=4,
            max_depth=2,
            n_jobs=1,
            random_state=42,
        )
        classifier.fit(features, labels, verbose=False)
        classifier.get_booster().feature_names = list(CANONICAL_FEATURE_ORDER)
        result = verify_probability_parity(
            classifier,
            tuple(np.median(features, axis=0)),
            features,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["maximum_absolute_difference"], 0.0)
        self.assertEqual(result["decision_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
