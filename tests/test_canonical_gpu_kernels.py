"""Focused CPU-compatibility and CUDA-dispatch tests for canonical kernels."""

from __future__ import annotations

import contextlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class CanonicalTorchDeviceTests(unittest.TestCase):
    @staticmethod
    def _fake_torch(cuda_available: bool) -> types.ModuleType:
        module = types.ModuleType("torch")
        module.float32 = object()
        module.cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available,
            current_device=lambda: 0,
        )
        return module

    def test_sam2_cuda_is_explicit_and_never_falls_back(self) -> None:
        from compag_curation.canonical import model_loading

        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
                model_loading._resolve_torch_device("cpu")
            with self.assertRaisesRegex(ValueError, "CUDA-only"):
                model_loading._resolve_torch_device("auto")
        with mock.patch.dict(
            sys.modules,
            {"torch": self._fake_torch(False)},
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot access a CUDA device"):
                model_loading._resolve_torch_device("cuda")

    def test_sam2_builder_receives_resolved_cuda_device(self) -> None:
        from compag_curation.canonical import model_loading
        from compag_curation.canonical.spec import (
            CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
            CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
            CanonicalAMGSettings,
        )

        observed: dict[str, object] = {}
        fake_sam2 = types.ModuleType("sam2")
        fake_build_module = types.ModuleType("sam2.build_sam")

        def fake_build(config_name, checkpoint_path, **kwargs):
            observed["config_name"] = config_name
            observed["checkpoint_path"] = checkpoint_path
            observed["kwargs"] = kwargs
            return "cuda-model"

        fake_build_module.build_sam2 = fake_build

        @contextlib.contextmanager
        def fake_verified(path, *_args, **_kwargs):
            yield Path(path), object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_init = root / "sam2/__init__.py"
            package_init.parent.mkdir()
            package_init.write_bytes(b"")
            fake_sam2.__file__ = str(package_init)
            supplied = root / "sam2.yaml"
            checkpoint = root / "sam2.pt"
            supplied.write_bytes(b"model: {}\n")
            checkpoint.write_bytes(b"checkpoint")
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "sam2": fake_sam2,
                        "sam2.build_sam": fake_build_module,
                        "torch": self._fake_torch(True),
                    },
                ),
                mock.patch.object(
                    model_loading,
                    "verified_file_path",
                    side_effect=fake_verified,
                ),
                mock.patch.object(
                    model_loading,
                    "_register_verified_sam2_config",
                    return_value="verified-config",
                ),
                mock.patch.object(
                    model_loading,
                    "_attest_canonical_sam2_cuda_model",
                    return_value=0,
                ),
            ):
                model = model_loading.load_canonical_sam2_model(
                    supplied,
                    checkpoint,
                    CanonicalAMGSettings().config_locator,
                    CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
                    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
                    device="cuda",
                )

        self.assertEqual(model, "cuda-model")
        self.assertEqual(observed["kwargs"]["device"], "cuda")
        self.assertEqual(observed["kwargs"]["mode"], "eval")
        self.assertFalse(observed["kwargs"]["apply_postprocessing"])

    def test_resnet_defaults_to_cuda_and_rejects_cpu_before_model_work(self) -> None:
        from compag_curation.canonical import features
        from compag_curation.canonical.features import (
            CANONICAL_RESNET50_GPU_BATCH_SIZE,
            CanonicalResNet50Embedder,
        )

        class Model:
            def __init__(self) -> None:
                self.calls: list[object] = []

            def float(self):
                self.calls.append("float")
                return self

            def to(self, device):
                self.calls.append(("to", device))
                return self

            def eval(self):
                self.calls.append("eval")
                return self

        with mock.patch.dict(
            sys.modules,
            {"torch": self._fake_torch(True)},
        ), mock.patch.object(features, "_attest_cuda_module"):
            gpu_model = Model()
            gpu = CanonicalResNet50Embedder(
                gpu_model,
                lambda value: value,
            )

        self.assertEqual(gpu.device, "cuda")
        self.assertEqual(gpu.batch_size, CANONICAL_RESNET50_GPU_BATCH_SIZE)
        self.assertEqual(gpu_model.calls, ["float", ("to", "cuda:0"), "eval"])
        cpu_model = Model()
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
                CanonicalResNet50Embedder(
                    cpu_model,
                    lambda value: value,
                    device="cpu",
                )
        self.assertEqual(cpu_model.calls, [])

    def test_resnet_local_loader_rejects_cpu_before_assets_or_model_loading(self) -> None:
        from compag_curation.canonical import features

        with mock.patch.object(
            features,
            "verified_file_path",
            side_effect=AssertionError("asset verification must not run"),
        ) as verified:
            with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
                features.CanonicalResNet50Embedder.from_local_weights(
                    Path("weights.pth"),
                    "0" * 64,
                    device="cpu",
                )
        verified.assert_not_called()

    def test_resnet_per_call_attests_state_input_and_raw_output(self) -> None:
        from compag_curation.canonical import features

        fake_torch = self._fake_torch(True)

        class Device:
            def __init__(self, kind: str) -> None:
                self.type = kind
                self.index = 0 if kind == "cuda" else None

        class Tensor:
            def __init__(self, values, *, kind: str = "cuda", dtype=None) -> None:
                self.values = np.asarray(values)
                self.device = Device(kind)
                self.dtype = fake_torch.float32 if dtype is None else dtype

            def is_floating_point(self) -> bool:
                return True

            def to(self, **_kwargs):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class Model:
            def __init__(self, output: Tensor) -> None:
                self.output = output
                self.calls = 0

            def float(self):
                return self

            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, _inputs):
                self.calls += 1
                return self.output

        fake_torch.inference_mode = contextlib.nullcontext
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.COLOR_BGR2RGB = object()
        fake_cv2.cvtColor = lambda value, _conversion: value

        def run(input_tensor: Tensor, output_tensor: Tensor):
            fake_torch.stack = lambda _values, *, dim: input_tensor
            model = Model(output_tensor)
            with (
                mock.patch.dict(
                    sys.modules,
                    {"cv2": fake_cv2, "torch": fake_torch},
                ),
                mock.patch.object(features, "_attest_cuda_module") as state_attest,
                mock.patch.object(
                    features,
                    "_attest_cuda_float32_tensor",
                    wraps=features._attest_cuda_float32_tensor,
                ) as tensor_attest,
            ):
                embedder = features.CanonicalResNet50Embedder(
                    model,
                    lambda _rgb: object(),
                )
                result = embedder.embed_many(
                    (np.zeros((2, 2, 3), dtype=np.uint8),)
                )
            return result, model, state_attest, tensor_attest

        values = np.arange(1, 2049, dtype=np.float32).reshape(1, -1)
        result, model, state_attest, tensor_attest = run(
            Tensor(np.zeros((1, 3, 2, 2), dtype=np.float32)),
            Tensor(values),
        )
        self.assertEqual(result.shape, (1, 2048))
        self.assertEqual(model.calls, 1)
        self.assertEqual(state_attest.call_count, 3)
        self.assertEqual(
            [call.kwargs["role"] for call in tensor_attest.call_args_list],
            [
                "canonical ResNet50 input batch",
                "canonical ResNet50 output batch",
            ],
        )

        for role, input_tensor, output_tensor, message in (
            (
                "input-device",
                Tensor(np.zeros((1, 3, 2, 2)), kind="cpu"),
                Tensor(values),
                "input batch is not on cuda:0",
            ),
            (
                "output-device",
                Tensor(np.zeros((1, 3, 2, 2))),
                Tensor(values, kind="cpu"),
                "output batch is not on cuda:0",
            ),
            (
                "output-dtype",
                Tensor(np.zeros((1, 3, 2, 2))),
                Tensor(values, dtype=object()),
                "output batch is not FP32",
            ),
        ):
            with self.subTest(role=role):
                with self.assertRaisesRegex(RuntimeError, message):
                    run(input_tensor, output_tensor)

    def test_sam2_attestation_closes_tensor_dtype_eval_and_device_state(self) -> None:
        from compag_curation.canonical import model_loading

        fake_torch = self._fake_torch(True)

        class Device:
            def __init__(self, kind: str, index: int | None) -> None:
                self.type = kind
                self.index = index

        class Tensor:
            def __init__(self, kind: str, *, dtype=None, floating: bool = True) -> None:
                self.device = Device(kind, 0 if kind == "cuda" else None)
                self.dtype = fake_torch.float32 if dtype is None else dtype
                self.floating = floating

            def is_floating_point(self) -> bool:
                return self.floating

        class Model:
            def __init__(
                self,
                *,
                parameters=(),
                buffers=(),
                training: bool = False,
                model_kind: str = "cuda",
            ) -> None:
                self.device = Device(
                    model_kind,
                    0 if model_kind == "cuda" else None,
                )
                self.parameters = tuple(parameters)
                self.buffers = tuple(buffers)
                self.training = training

            def named_parameters(self, *, recurse: bool):
                self.assert_recurse(recurse)
                return tuple(
                    (f"parameter-{index}", tensor)
                    for index, tensor in enumerate(self.parameters)
                )

            def named_buffers(self, *, recurse: bool):
                self.assert_recurse(recurse)
                return tuple(
                    (f"buffer-{index}", tensor)
                    for index, tensor in enumerate(self.buffers)
                )

            def modules(self):
                return (types.SimpleNamespace(training=self.training),)

            @staticmethod
            def assert_recurse(recurse: bool) -> None:
                if not recurse:
                    raise AssertionError("attestation must recurse")

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            valid = Model(
                parameters=(Tensor("cuda"),),
                buffers=(Tensor("cuda", dtype=object(), floating=False),),
            )
            self.assertEqual(
                model_loading._attest_canonical_sam2_cuda_model(valid),
                0,
            )
            cases = (
                (
                    "parameterless",
                    Model(),
                    "has no parameters",
                ),
                (
                    "cpu-buffer",
                    Model(
                        parameters=(Tensor("cuda"),),
                        buffers=(Tensor("cpu"),),
                    ),
                    "tensor is not on cuda:0",
                ),
                (
                    "fp16",
                    Model(
                        parameters=(Tensor("cuda", dtype=object()),),
                    ),
                    "floating tensor is not FP32",
                ),
                (
                    "training",
                    Model(parameters=(Tensor("cuda"),), training=True),
                    "not in eval mode",
                ),
                (
                    "model-device",
                    Model(parameters=(Tensor("cuda"),), model_kind="cpu"),
                    "model.device is not cuda:0",
                ),
            )
            for role, model, message in cases:
                with self.subTest(role=role):
                    with self.assertRaisesRegex(RuntimeError, message):
                        model_loading._attest_canonical_sam2_cuda_model(model)

    def test_amg_builder_attests_model_before_importing_upstream_generator(self) -> None:
        from compag_curation.canonical import model_loading, proposals

        with (
            mock.patch.object(
                model_loading,
                "_attest_canonical_sam2_cuda_model",
                side_effect=RuntimeError("CPU model rejected"),
            ) as attestation,
            mock.patch.dict(
                sys.modules,
                {"sam2.automatic_mask_generator": None},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU model rejected"):
                proposals.build_canonical_amg(mock.sentinel.cpu_model)
        attestation.assert_called_once_with(mock.sentinel.cpu_model)

    def test_amg_builder_seals_method_and_execution_microbatches(self) -> None:
        from compag_curation.canonical import model_loading, proposals
        from compag_curation.canonical.spec import (
            CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
        )

        captured: list[dict[str, object]] = []
        automatic_mask_generator = types.ModuleType("sam2.automatic_mask_generator")

        class Generator:
            def __init__(self, _model: object, **kwargs: object) -> None:
                captured.append(dict(kwargs))

        automatic_mask_generator.SAM2AutomaticMaskGenerator = Generator
        sam2 = types.ModuleType("sam2")
        with (
            mock.patch.object(
                model_loading,
                "_attest_canonical_sam2_cuda_model",
                return_value=0,
            ),
            mock.patch.object(
                proposals,
                "_attest_canonical_amg_cuda",
                return_value=0,
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "sam2": sam2,
                    "sam2.automatic_mask_generator": automatic_mask_generator,
                },
            ),
        ):
            proposals.build_canonical_amg(mock.sentinel.model)
            proposals.build_canonical_amg(
                mock.sentinel.model,
                execution_points_per_batch=CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            )
            proposals.build_canonical_amg(
                mock.sentinel.model,
                execution_points_per_batch=CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            )
            for unsupported in (True, 64, 128, 1024):
                with self.subTest(unsupported=unsupported), self.assertRaisesRegex(
                    ValueError,
                    "microbatch",
                ):
                    proposals.build_canonical_amg(
                        mock.sentinel.model,
                        execution_points_per_batch=unsupported,
                    )

        self.assertEqual(
            [record["points_per_batch"] for record in captured],
            [
                512,
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
                CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            ],
        )
        self.assertEqual(CANONICAL_STAGE20_AMG_POINTS_PER_BATCH, 32)
        self.assertEqual(CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH, 32)
        self.assertTrue(all(record["points_per_side"] == 64 for record in captured))

    def test_amg_memory_wrapper_drops_only_unused_non_m2m_logits(self) -> None:
        from compag_curation.canonical import proposals

        low_res = object()
        masks = object()
        scores = object()

        class BatchData:
            def __init__(self) -> None:
                self.values = {
                    "low_res_masks": low_res,
                    "masks": masks,
                    "iou_preds": scores,
                }

            def items(self):
                return self.values.items()

            def __delitem__(self, key: str) -> None:
                del self.values[key]

        class Upstream:
            def __init__(self, *, use_m2m: bool) -> None:
                self.use_m2m = use_m2m
                self.returned = BatchData()

            def _process_batch(self, *_args, **_kwargs):
                return self.returned

        bounded = proposals._memory_bounded_sam2_amg_type(Upstream)
        without_m2m = bounded(use_m2m=False)
        observed = without_m2m._process_batch(mock.sentinel.points)
        self.assertIs(observed, without_m2m.returned)
        self.assertEqual(set(observed.values), {"masks", "iou_preds"})
        self.assertIs(observed.values["masks"], masks)
        self.assertIs(observed.values["iou_preds"], scores)

        with_m2m = bounded(use_m2m=True)
        observed = with_m2m._process_batch(mock.sentinel.points)
        self.assertIs(observed, with_m2m.returned)
        self.assertEqual(
            set(observed.values),
            {"low_res_masks", "masks", "iou_preds"},
        )
        self.assertIs(observed.values["low_res_masks"], low_res)
        self.assertIs(observed.values["masks"], masks)
        self.assertIs(observed.values["iou_preds"], scores)

    def test_amg_attestation_rejects_replaced_model_and_wrong_predictor_device(
        self,
    ) -> None:
        from compag_curation.canonical import model_loading, proposals

        class Device:
            def __init__(self, kind: str) -> None:
                self.type = kind
                self.index = 0 if kind == "cuda" else None

        model = object()
        valid = types.SimpleNamespace(
            predictor=types.SimpleNamespace(
                model=model,
                device=Device("cuda"),
            )
        )
        with mock.patch.object(
            model_loading,
            "_attest_canonical_sam2_cuda_model",
            return_value=0,
        ):
            self.assertEqual(
                proposals._attest_canonical_amg_cuda(
                    valid,
                    expected_model=model,
                ),
                0,
            )
            with self.assertRaisesRegex(RuntimeError, "replaced"):
                proposals._attest_canonical_amg_cuda(
                    valid,
                    expected_model=object(),
                )
            valid.predictor.device = Device("cpu")
            with self.assertRaisesRegex(RuntimeError, "predictor.device"):
                proposals._attest_canonical_amg_cuda(valid)


class CanonicalXGBoostDeviceTests(unittest.TestCase):
    def test_training_policy_is_cuda_only_and_bounds_host_threads(self) -> None:
        from compag_curation.canonical.training import (
            CANONICAL_XGB_GPU_MAX_HOST_THREADS,
            CanonicalTrainingConfig,
            _base_classifier_parameters,
        )

        cuda = CanonicalTrainingConfig()
        parameters = _base_classifier_parameters(cuda)
        self.assertEqual(cuda.device, "cuda")
        self.assertEqual(parameters["tree_method"], "hist")
        self.assertEqual(parameters["device"], "cuda")
        self.assertEqual(parameters["n_jobs"], 1)
        self.assertIs(parameters["fail_on_invalid_gpu_id"], True)
        with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
            CanonicalTrainingConfig(device="cpu")
        with self.assertRaisesRegex(ValueError, "sealed at 1"):
            CanonicalTrainingConfig(
                device="cuda",
                thread_count=CANONICAL_XGB_GPU_MAX_HOST_THREADS + 1,
            )

    def test_gpu_training_config_is_host_count_independent(self) -> None:
        from compag_curation.canonical import service

        config = types.SimpleNamespace(device="cuda")
        with (
            mock.patch.object(service, "_canonical_config"),
            mock.patch.object(
                service.os,
                "cpu_count",
                side_effect=AssertionError("host CPU count must not be observed"),
            ) as cpu_count,
        ):
            training = service._canonical_training_config(config)
        self.assertEqual(training.thread_count, 1)
        cpu_count.assert_not_called()

    @staticmethod
    def _fake_cuda_modules():
        from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER

        observed: dict[str, list[object]] = {
            "cuda_inputs": [],
            "dmatrix_inputs": [],
            "parameters": [],
        }

        class CudaArray:
            def __init__(self, value):
                self.value = np.asarray(value)

            @property
            def __cuda_array_interface__(self):
                return {
                    "shape": self.value.shape,
                    "typestr": self.value.dtype.str,
                    "data": (1, False),
                    "version": 3,
                }

        cupy = types.ModuleType("cupy")
        cupy.cuda = types.SimpleNamespace(
            runtime=types.SimpleNamespace(
                getDeviceCount=lambda: 1,
                getDevice=lambda: 0,
            )
        )

        def asarray(value):
            converted = CudaArray(value)
            observed["cuda_inputs"].append(converted)
            return converted

        cupy.asarray = asarray
        cupy.asnumpy = lambda value: np.asarray(value.value)

        class DMatrix:
            def __init__(self, value, *, feature_names):
                self.value = value
                self.feature_names = feature_names
                observed["dmatrix_inputs"].append(value)

        class Booster:
            feature_names = list(CANONICAL_FEATURE_ORDER)

            def __init__(self) -> None:
                self.generic = {
                    "device": "cuda:0",
                    "nthread": "1",
                    "fail_on_invalid_gpu_id": "1",
                }

            def load_model(self, _payload) -> None:
                return None

            def num_features(self) -> int:
                return len(CANONICAL_FEATURE_ORDER)

            def set_param(self, parameters) -> None:
                observed["parameters"].append(dict(parameters))
                self.generic.update(
                    {
                        "device": (
                            "cuda:0"
                            if parameters.get("device") == "cuda"
                            else parameters.get("device", self.generic["device"])
                        ),
                        "nthread": str(
                            parameters.get("nthread", self.generic["nthread"])
                        ),
                        "fail_on_invalid_gpu_id": (
                            "1"
                            if parameters.get("fail_on_invalid_gpu_id") is True
                            else self.generic["fail_on_invalid_gpu_id"]
                        ),
                    }
                )

            def save_config(self) -> str:
                import json

                return json.dumps({"learner": {"generic_param": self.generic}})

            def inplace_predict(self, matrix, *, validate_features):
                if validate_features:
                    raise AssertionError("array prediction unexpectedly validated names")
                return CudaArray(
                    np.full(len(matrix.value), 0.25, dtype=np.float32)
                )

        xgboost = types.ModuleType("xgboost")
        xgboost.build_info = lambda: {"USE_CUDA": True}
        xgboost.Booster = Booster
        xgboost.DMatrix = DMatrix
        xgboost.core = types.SimpleNamespace(XGBoostError=RuntimeError)
        return xgboost, cupy, observed

    def test_portable_cuda_predictor_stores_device_and_uses_cupy(self) -> None:
        from compag_curation.canonical.serialization import (
            load_portable_predictor,
            predict_portable_probabilities,
        )
        from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER

        xgboost, cupy, observed = self._fake_cuda_modules()
        with mock.patch.dict(
            sys.modules,
            {"xgboost": xgboost, "cupy": cupy},
        ):
            predictor = load_portable_predictor(
                b"fixture-ubj",
                np.zeros(len(CANONICAL_FEATURE_ORDER), dtype=np.float32),
                device="cuda",
            )
            probabilities = predict_portable_probabilities(
                predictor,
                np.zeros((3, len(CANONICAL_FEATURE_ORDER)), dtype=np.float32),
            )

        self.assertEqual(predictor.device, "cuda")
        expected_parameters = {
            "device": "cuda",
            "nthread": 1,
            "fail_on_invalid_gpu_id": True,
        }
        self.assertEqual(
            observed["parameters"],
            [expected_parameters, expected_parameters],
        )
        self.assertEqual(len(observed["cuda_inputs"]), 1)
        self.assertTrue(np.array_equal(probabilities, np.full(3, 0.25, np.float32)))

    def test_portable_cuda_predictor_rejects_cpu_only_xgboost_build(self) -> None:
        from compag_curation.canonical.serialization import load_portable_predictor

        xgboost = types.ModuleType("xgboost")
        xgboost.build_info = lambda: {"USE_CUDA": False}
        with mock.patch.dict(sys.modules, {"xgboost": xgboost}):
            with self.assertRaisesRegex(RuntimeError, "no CUDA support"):
                load_portable_predictor(b"fixture-ubj", [0.0] * 93, device="cuda")

    def test_cpu_public_training_and_portable_apis_reject_before_imports_or_work(self) -> None:
        from compag_curation.canonical.serialization import (
            load_portable_predictor,
            verify_probability_parity,
        )
        from compag_curation.canonical.training import train_canonical_xgb

        cpu_config = types.SimpleNamespace(device="cpu")
        with mock.patch.dict(sys.modules, {"numpy": None, "xgboost": None}):
            with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
                train_canonical_xgb([], [], [], [], [], config=cpu_config)
            with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
                load_portable_predictor(b"ubj", (), device="cpu")
            with self.assertRaisesRegex(ValueError, "CUDA-only.*CPU"):
                verify_probability_parity(object(), (), object(), device="cpu")

    def test_effective_booster_config_rejects_silent_cpu_fallback(self) -> None:
        import json

        from compag_curation.canonical.training import (
            _attest_xgboost_cuda_booster,
        )

        class Booster:
            def save_config(self) -> str:
                return json.dumps(
                    {
                        "learner": {
                            "generic_param": {
                                "device": "cpu",
                                "nthread": "1",
                                "fail_on_invalid_gpu_id": "1",
                            }
                        }
                    }
                )

        cp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDevice=lambda: 0)
            )
        )
        with self.assertRaisesRegex(RuntimeError, "effective CUDA"):
            _attest_xgboost_cuda_booster(Booster(), cp)

    def test_cuda_prediction_rejects_host_output(self) -> None:
        from compag_curation.canonical.training import _positive_probabilities

        xgboost, cupy, _observed = self._fake_cuda_modules()

        class Model:
            def __init__(self) -> None:
                self.booster = xgboost.Booster()

            def get_booster(self):
                return self.booster

        model = Model()
        model.booster.inplace_predict = lambda *_args, **_kwargs: np.full(
            2,
            0.5,
            dtype=np.float32,
        )
        with mock.patch.dict(sys.modules, {"xgboost": xgboost, "cupy": cupy}):
            with self.assertRaisesRegex(RuntimeError, "host array"):
                _positive_probabilities(
                    model,
                    np.zeros((2, 3), dtype=np.float32),
                    device="cuda",
                )

    def test_xgboost_device_fallback_warning_is_a_hard_failure(self) -> None:
        import json
        import warnings

        from compag_curation.canonical.training import (
            _attest_xgboost_cuda_booster,
        )

        class Booster:
            def save_config(self) -> str:
                warnings.warn(
                    "No visible GPU is found, setting device to CPU",
                    UserWarning,
                )
                return json.dumps(
                    {
                        "learner": {
                            "generic_param": {
                                "device": "cpu",
                                "nthread": "1",
                                "fail_on_invalid_gpu_id": "1",
                            }
                        }
                    }
                )

        cp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDevice=lambda: 0)
            )
        )
        with self.assertRaisesRegex(RuntimeError, "fallback warning"):
            _attest_xgboost_cuda_booster(Booster(), cp)


if __name__ == "__main__":
    unittest.main()
