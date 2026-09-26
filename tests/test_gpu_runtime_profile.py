from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class _FakeTensor:
    def __init__(self, index: int, value: float = 1.0) -> None:
        self.is_cuda = True
        self.device = SimpleNamespace(type="cuda", index=index)
        self.value = value

    def __matmul__(self, other: object) -> _FakeTensor:
        return _FakeTensor(self.device.index, 2.0)

    def sum(self) -> _FakeTensor:
        return _FakeTensor(self.device.index, 8.0)

    def item(self) -> float:
        return self.value


class _FakeCudnn:
    benchmark = True
    deterministic = False
    allow_tf32 = True

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def version() -> int:
        return 92000


class _FakeMatmul:
    allow_tf32 = True


class _FakeCudaBackend:
    matmul = _FakeMatmul


class _FakeCuda:
    available = True
    initialized = True
    current = 0
    synchronized: list[int] = []
    allocated = 256
    reserved = 512

    @classmethod
    def is_available(cls) -> bool:
        return cls.available

    @classmethod
    def is_initialized(cls) -> bool:
        return cls.initialized

    @staticmethod
    def device_count() -> int:
        return 2

    @classmethod
    def set_device(cls, index: int) -> None:
        cls.current = index

    @classmethod
    def current_device(cls) -> int:
        return cls.current

    @staticmethod
    def get_device_properties(index: int) -> object:
        return SimpleNamespace(name=f"Fixture GPU {index}", total_memory=24 * 1024**3)

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        return (8, 9)

    @classmethod
    def synchronize(cls, index: int) -> None:
        cls.synchronized.append(index)

    @classmethod
    def memory_allocated(cls, index: int) -> int:
        return cls.allocated

    @classmethod
    def memory_reserved(cls, index: int) -> int:
        return cls.reserved

    @classmethod
    def empty_cache(cls) -> None:
        cls.allocated = 0
        cls.reserved = 0


class _FakeTorchInternal:
    clear_cublas_workspaces_calls = 0

    @classmethod
    def _cuda_clearCublasWorkspaces(cls) -> None:
        cls.clear_cublas_workspaces_calls += 1


class _FakeTorch:
    version = SimpleNamespace(cuda="13.2")
    cuda = _FakeCuda
    _C = _FakeTorchInternal
    backends = SimpleNamespace(cudnn=_FakeCudnn, cuda=_FakeCudaBackend)
    float32 = object()
    deterministic = False
    matmul_precision = "medium"
    default_dtype: object | None = None
    autocast_enabled = False
    num_threads = 8
    interop_threads = 8

    @classmethod
    def use_deterministic_algorithms(cls, enabled: bool) -> None:
        cls.deterministic = enabled

    @classmethod
    def are_deterministic_algorithms_enabled(cls) -> bool:
        return cls.deterministic

    @classmethod
    def set_float32_matmul_precision(cls, value: str) -> None:
        cls.matmul_precision = value

    @classmethod
    def get_float32_matmul_precision(cls) -> str:
        return cls.matmul_precision

    @classmethod
    def set_default_dtype(cls, value: object) -> None:
        cls.default_dtype = value

    @classmethod
    def get_default_dtype(cls) -> object:
        return cls.default_dtype

    @classmethod
    def is_autocast_enabled(cls, device_type: str) -> bool:
        assert device_type == "cuda"
        return cls.autocast_enabled

    @classmethod
    def set_num_threads(cls, threads: int) -> None:
        cls.num_threads = threads

    @classmethod
    def get_num_threads(cls) -> int:
        return cls.num_threads

    @classmethod
    def set_num_interop_threads(cls, threads: int) -> None:
        cls.interop_threads = threads

    @classmethod
    def get_num_interop_threads(cls) -> int:
        return cls.interop_threads

    @staticmethod
    def ones(shape: tuple[int, int], *, dtype: object, device: str) -> _FakeTensor:
        assert shape == (2, 2)
        assert dtype is _FakeTorch.float32
        return _FakeTensor(int(device.split(":", 1)[1]))


class _FakeXGBoost:
    @staticmethod
    def build_info() -> dict[str, object]:
        return {
            "CUDA_VERSION": [12, 8],
            "USE_CUDA": True,
            "USE_DLOPEN_NCCL": True,
            "USE_NCCL": True,
            "USE_OPENMP": True,
            "USE_RMM": False,
            "libxgboost": "/unrecorded/environment/path/libxgboost.so",
        }


class _FakeCupyRuntime:
    @staticmethod
    def driverGetVersion() -> int:
        return 13020


class _FakeCupy:
    cuda = SimpleNamespace(runtime=_FakeCupyRuntime)


class _FakeOcl:
    enabled = True

    @classmethod
    def setUseOpenCL(cls, enabled: bool) -> None:
        cls.enabled = enabled

    @classmethod
    def useOpenCL(cls) -> bool:
        return cls.enabled


class _FakeCv2:
    threads = 8
    ocl = _FakeOcl

    @classmethod
    def setNumThreads(cls, threads: int) -> None:
        cls.threads = threads

    @classmethod
    def getNumThreads(cls) -> int:
        return cls.threads


class _FakeThreadpoolctl:
    threads = 8

    @classmethod
    def threadpool_limits(cls, *, limits: int) -> object:
        cls.threads = limits
        return object()

    @classmethod
    def threadpool_info(cls) -> list[dict[str, object]]:
        return [{"internal_api": "fixture", "num_threads": cls.threads}]


_FAKE_SAM2_ATTESTATION = {
    "schema": "compag-curation-sam2-cuda-extension-attestation/v1",
    "sha256": "a" * 64,
    "probe": {"cpu_fallback": False},
}


class _FakeArchiveDistribution:
    def __init__(self, name: str, row: dict[str, str], metadata_path: Path) -> None:
        self.metadata = {"Name": name}
        self.version = row["version"]
        self._path = metadata_path
        self.direct_text: str | None = json.dumps(
            {
                "archive_info": {
                    "hash": f"sha256={row['sha256']}",
                    "hashes": {"sha256": row["sha256"]},
                },
                "url": row["url"],
            },
            sort_keys=True,
        )

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json":
            raise AssertionError(name)
        return self.direct_text


def _archive_distribution_fixtures(
    archives: dict[str, dict[str, str]],
    root: Path,
) -> dict[str, _FakeArchiveDistribution]:
    fixtures: dict[str, _FakeArchiveDistribution] = {}
    for name, row in archives.items():
        metadata_path = root / f"{name.replace('-', '_')}-{row['version']}.dist-info"
        metadata_path.mkdir()
        fixtures[name] = _FakeArchiveDistribution(name, row, metadata_path)
    return fixtures


@contextmanager
def _archive_fixture_environment(public_backend: object, archives: dict[str, dict[str, str]]):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve(strict=True)
        fixtures = _archive_distribution_fixtures(archives, root)
        with mock.patch.object(
            public_backend,
            "_science_gpu_distribution_roots",
            return_value=(root,),
        ):
            yield fixtures, root


class _FakePool:
    def __init__(self, used: int, total: int, free_blocks: int = 0) -> None:
        self.used = used
        self.total = total
        self.free_blocks = free_blocks

    def used_bytes(self) -> int:
        return self.used

    def total_bytes(self) -> int:
        return self.total

    def n_free_blocks(self) -> int:
        return self.free_blocks

    def free_all_blocks(self) -> None:
        self.used = 0
        self.total = 0
        self.free_blocks = 0


class _FakeCupyDevice:
    synchronized: list[int] = []

    def __init__(self, index: int) -> None:
        self.index = index

    def synchronize(self) -> None:
        self.synchronized.append(self.index)


class ScienceGpuRuntimeTests(unittest.TestCase):
    def test_accelerator_cache_environment_uses_nonempty_runtime_root(self) -> None:
        from compag_curation.public_io import owned_manifest_contract, write_new_json
        from compag_curation.runtime_lock import confined_accelerator_cache_environment

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory).resolve() / "runtime"
            runtime.mkdir(mode=0o700)
            write_new_json(runtime / "ENVIRONMENT_POLICY.json", {"status": "PASS"})
            environment = confined_accelerator_cache_environment(runtime)
            self.assertEqual(
                set(environment),
                {
                    "CUDA_CACHE_PATH",
                    "CUPY_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                },
            )
            self.assertEqual(set(environment.values()), {str(runtime)})
            for configured in environment.values():
                Path(configured).mkdir(mode=0o700, exist_ok=True)
            files, directories = owned_manifest_contract(runtime)
            self.assertEqual(
                [row["path"] for row in files],
                ["ENVIRONMENT_POLICY.json"],
            )
            self.assertEqual(directories, [{"path": ".", "mode_octal": "0700"}])

        with self.assertRaisesRegex(RuntimeError, "must be absolute"):
            confined_accelerator_cache_environment(Path("relative-runtime"))

    def test_runtime_cleanup_prunes_only_owned_empty_descendants(self) -> None:
        from compag_curation.public_io import (
            PublicIOError,
            prune_owned_empty_directories,
            write_new_bytes,
        )

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir(mode=0o700)
            write_new_bytes(runtime / "ENVIRONMENT_POLICY.json", b"evidence\n")
            empty_leaf = runtime / "torchinductor_fixture/nested"
            empty_leaf.mkdir(parents=True)
            nonempty = runtime / ".cupy/kernel_cache"
            nonempty.mkdir(parents=True)
            write_new_bytes(nonempty / "kernel.cubin", b"compiled\n")
            removed = prune_owned_empty_directories(runtime)
            self.assertEqual(
                removed,
                ("torchinductor_fixture", "torchinductor_fixture/nested"),
            )
            self.assertFalse((runtime / "torchinductor_fixture").exists())
            self.assertEqual((nonempty / "kernel.cubin").read_bytes(), b"compiled\n")
            self.assertEqual(
                (runtime / "ENVIRONMENT_POLICY.json").read_bytes(),
                b"evidence\n",
            )

            unsafe = runtime / "unsafe-link"
            unsafe.symlink_to(nonempty, target_is_directory=True)
            with self.assertRaisesRegex(PublicIOError, "unsafe directory"):
                prune_owned_empty_directories(runtime)

    def test_runtime_cache_purge_retains_only_named_evidence(self) -> None:
        from compag_curation.public_io import (
            PublicIOError,
            owned_manifest_contract,
            purge_owned_runtime_cache_files,
            write_new_bytes,
        )

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir(mode=0o700)
            write_new_bytes(runtime / "ENVIRONMENT_POLICY.json", b"policy\n")
            write_new_bytes(
                runtime / "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
                b"release\n",
            )
            triton = runtime / "9/0"
            triton.mkdir(parents=True, mode=0o700)
            write_new_bytes(triton / "c25eed8a829ad6", b"compiled\n")
            write_new_bytes(runtime / "kernel.cubin", b"cuda-cache\n")

            removed = purge_owned_runtime_cache_files(
                runtime,
                retained_files=(
                    "ENVIRONMENT_POLICY.json",
                    "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
                ),
            )
            self.assertEqual(removed, ("9/0/c25eed8a829ad6", "kernel.cubin"))
            files, directories = owned_manifest_contract(runtime)
            self.assertEqual(
                [row["path"] for row in files],
                [
                    "ENVIRONMENT_POLICY.json",
                    "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
                ],
            )
            self.assertEqual(directories, [{"path": ".", "mode_octal": "0700"}])

            unsafe = runtime / "unsafe-link"
            unsafe.symlink_to(runtime / "ENVIRONMENT_POLICY.json")
            with self.assertRaisesRegex(PublicIOError, "unsafe file"):
                purge_owned_runtime_cache_files(
                    runtime,
                    retained_files=("ENVIRONMENT_POLICY.json",),
                )

    def test_stage60_runtime_receipt_rejects_generated_cache_artifacts(self) -> None:
        from compag_curation.public_io import PublicIOError
        from compag_curation.public_pipeline import (
            _validated_stage_sixty_runtime_evidence,
        )

        receipt: dict[str, object] = {
            "files": [
                {"path": "fresh_process_runtime/ENVIRONMENT_POLICY.json"},
                {
                    "path": "fresh_process_runtime/GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json"
                },
                {"path": "predictions.csv"},
            ],
            "directories": [
                {"path": "."},
                {"path": "fresh_process_runtime"},
            ],
        }
        self.assertEqual(
            _validated_stage_sixty_runtime_evidence(receipt, device="cuda"),
            {
                "files": (
                    "fresh_process_runtime/ENVIRONMENT_POLICY.json",
                    "fresh_process_runtime/GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
                ),
                "directories": ("fresh_process_runtime",),
            },
        )
        receipt["files"] = [
            *receipt["files"],
            {"path": "fresh_process_runtime/9/0/c25eed8a829ad6"},
        ]
        receipt["directories"] = [
            *receipt["directories"],
            {"path": "fresh_process_runtime/9"},
            {"path": "fresh_process_runtime/9/0"},
        ]
        with self.assertRaisesRegex(PublicIOError, "generated cache artifacts"):
            _validated_stage_sixty_runtime_evidence(receipt, device="cuda")

    def test_runtime_cache_purge_rejects_unsafe_nodes_without_touching_targets(self) -> None:
        from compag_curation.public_io import (
            PublicIOError,
            purge_owned_runtime_cache_files,
            write_new_bytes,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir(mode=0o700)
            write_new_bytes(runtime / "ENVIRONMENT_POLICY.json", b"policy\n")
            target = root / "outside.txt"
            target.write_bytes(b"outside\n")

            symlink = runtime / "unsafe-symlink"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(PublicIOError, "unsafe file"):
                purge_owned_runtime_cache_files(
                    runtime,
                    retained_files=("ENVIRONMENT_POLICY.json",),
                )
            self.assertEqual(target.read_bytes(), b"outside\n")
            symlink.unlink()

            hardlink = runtime / "unsafe-hardlink"
            os.link(target, hardlink)
            with self.assertRaisesRegex(PublicIOError, "unsafe file"):
                purge_owned_runtime_cache_files(
                    runtime,
                    retained_files=("ENVIRONMENT_POLICY.json",),
                )
            self.assertEqual(target.read_bytes(), b"outside\n")
            hardlink.unlink()

            fifo = runtime / "unsafe-fifo"
            os.mkfifo(fifo, mode=0o600)
            with self.assertRaisesRegex(PublicIOError, "unsafe file"):
                purge_owned_runtime_cache_files(
                    runtime,
                    retained_files=("ENVIRONMENT_POLICY.json",),
                )
            self.assertEqual(target.read_bytes(), b"outside\n")

    def test_runtime_cache_purge_detects_path_and_evidence_swaps(self) -> None:
        from compag_curation.public_io import (
            PublicIOError,
            purge_owned_runtime_cache_files,
            write_new_bytes,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime-path-swap"
            runtime.mkdir(mode=0o700)
            evidence = runtime / "ENVIRONMENT_POLICY.json"
            cache = runtime / "cache.bin"
            target = root / "outside.txt"
            write_new_bytes(evidence, b"policy\n")
            write_new_bytes(cache, b"cache\n")
            target.write_bytes(b"outside\n")
            original_open = os.open
            swapped = False

            def swap_cache_before_parent_open(
                path: object,
                flags: int,
                *args: object,
                **kwargs: object,
            ) -> int:
                nonlocal swapped
                if not swapped and Path(path) == runtime:
                    cache.unlink()
                    cache.symlink_to(target)
                    swapped = True
                return original_open(path, flags, *args, **kwargs)

            with (
                mock.patch(
                    "compag_curation.public_io.os.open",
                    side_effect=swap_cache_before_parent_open,
                ),
                self.assertRaisesRegex(PublicIOError, "changed before cleanup"),
            ):
                purge_owned_runtime_cache_files(
                    runtime,
                    retained_files=("ENVIRONMENT_POLICY.json",),
                )
            self.assertTrue(cache.is_symlink())
            self.assertEqual(target.read_bytes(), b"outside\n")

            cache.unlink()
            write_new_bytes(cache, b"cache\n")
            original_unlink = os.unlink
            replaced = False

            def replace_evidence_during_unlink(
                name: object,
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal replaced
                if not replaced:
                    evidence.write_bytes(b"replaced\n")
                    replaced = True
                original_unlink(name, *args, **kwargs)

            with (
                mock.patch(
                    "compag_curation.public_io.os.unlink",
                    side_effect=replace_evidence_during_unlink,
                ),
                self.assertRaisesRegex(PublicIOError, "evidence changed"),
            ):
                purge_owned_runtime_cache_files(
                    runtime,
                    retained_files=("ENVIRONMENT_POLICY.json",),
                )

    def setUp(self) -> None:
        environment = mock.patch.dict(
            os.environ,
            {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"},
            clear=False,
        )
        environment.start()
        self.addCleanup(environment.stop)
        _FakeCuda.available = True
        _FakeCuda.initialized = True
        _FakeCuda.current = 0
        _FakeCuda.synchronized = []
        _FakeCuda.allocated = 256
        _FakeCuda.reserved = 512
        _FakeTorchInternal.clear_cublas_workspaces_calls = 0
        _FakeCupyDevice.synchronized = []
        _FakeCudnn.benchmark = True
        _FakeCudnn.deterministic = False
        _FakeCudnn.allow_tf32 = True
        _FakeMatmul.allow_tf32 = True
        _FakeTorch.deterministic = False
        _FakeTorch.matmul_precision = "medium"
        _FakeTorch.default_dtype = None
        _FakeTorch.autocast_enabled = False
        _FakeTorch.num_threads = 8
        _FakeTorch.interop_threads = 8
        _FakeCv2.threads = 8
        _FakeOcl.enabled = True
        _FakeThreadpoolctl.threads = 8

    def test_policy_is_deterministic_fp32_without_fallback(self) -> None:
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        policy = science_gpu_runtime_policy()
        self.assertEqual(policy["device"], "cuda")
        self.assertIs(policy["allow_cpu_fallback"], False)
        self.assertEqual(policy["precision"], "float32")
        self.assertIs(policy["autocast"], False)
        self.assertEqual(
            policy["environment"],
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            },
        )
        self.assertEqual(policy["native_threadpool_limit"], 1)
        self.assertEqual(policy["opencv_threads"], 1)
        self.assertIs(policy["opencv_opencl"], False)
        self.assertEqual(policy["torch_intraop_threads"], 1)
        self.assertEqual(policy["torch_interop_threads"], 1)
        self.assertIs(policy["torch_deterministic_algorithms"], True)
        self.assertIs(policy["cudnn_benchmark"], False)
        self.assertIs(policy["cudnn_allow_tf32"], False)
        self.assertIs(policy["cuda_matmul_allow_tf32"], False)
        self.assertRegex(str(policy["sha256"]), r"^[0-9a-f]{64}$")

    def test_archive_provenance_receipt_closes_all_62_archives(self) -> None:
        from compag_curation import public_backend
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
            with mock.patch.object(
                public_backend.importlib.metadata,
                "distributions",
                return_value=list(fixtures.values()),
            ):
                receipt = public_backend._science_gpu_archive_provenance_identity(
                    archives
                )
        self.assertEqual(
            receipt,
            {
                "schema": "compag-curation-science-gpu-archive-provenance/v1",
                "count": 62,
                "names": sorted(archives),
                "sha256": receipt["sha256"],
            },
        )
        self.assertRegex(str(receipt["sha256"]), r"^[0-9a-f]{64}$")
        self.assertNotIn("https://", repr(receipt))

        name = sorted(archives)[0]
        row = archives[name]
        for archive_info in (
            {"hashes": {"sha256": row["sha256"]}},
            {"hash": f"sha256={row['sha256']}"},
        ):
            with self.subTest(fields=sorted(archive_info)):
                with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
                    fixtures[name].direct_text = json.dumps(
                        {"archive_info": archive_info, "url": row["url"]},
                        sort_keys=True,
                    )
                    with mock.patch.object(
                        public_backend.importlib.metadata,
                        "distributions",
                        return_value=list(fixtures.values()),
                    ):
                        alternate = public_backend._science_gpu_archive_provenance_identity(
                            archives
                        )
                self.assertEqual(alternate, receipt)

    def test_archive_provenance_rejects_missing_direct_url(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
            fixtures[sorted(fixtures)[0]].direct_text = None
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=list(fixtures.values()),
                ),
                self.assertRaisesRegex(PublicIOError, "direct_url.json is missing"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

    def test_archive_provenance_rejects_wrong_url_and_sha256(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        name = sorted(archives)[0]
        row = archives[name]
        with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
            fixtures[name].direct_text = json.dumps(
                {
                    "archive_info": {
                        "hash": f"sha256={row['sha256']}",
                        "hashes": {"sha256": row["sha256"]},
                    },
                    "url": row["url"] + ".different",
                },
                sort_keys=True,
            )
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=list(fixtures.values()),
                ),
                self.assertRaisesRegex(PublicIOError, "URL differs"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

        wrong_sha256 = "0" * 64 if row["sha256"] != "0" * 64 else "1" * 64
        with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
            fixtures[name].direct_text = json.dumps(
                {
                    "archive_info": {
                        "hash": f"sha256={wrong_sha256}",
                        "hashes": {"sha256": wrong_sha256},
                    },
                    "url": row["url"],
                },
                sort_keys=True,
            )
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=list(fixtures.values()),
                ),
                self.assertRaisesRegex(PublicIOError, "SHA256 differs"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

    def test_archive_provenance_rejects_conda_path_and_vcs_origins(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        name = sorted(archives)[0]
        origins = (
            {
                "dir_info": {},
                "url": "".join(("file", chr(58), "///opt/conda/build_artifacts/package")),
            },
            {
                "url": "https://github.com/example/package.git",
                "vcs_info": {"commit_id": "0" * 40, "vcs": "git"},
            },
        )
        for origin in origins:
            with self.subTest(fields=sorted(origin)):
                with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
                    fixtures[name].direct_text = json.dumps(origin, sort_keys=True)
                    with (
                        mock.patch.object(
                            public_backend.importlib.metadata,
                            "distributions",
                            return_value=list(fixtures.values()),
                        ),
                        self.assertRaisesRegex(PublicIOError, "not an archive direct URL"),
                    ):
                        public_backend._science_gpu_archive_provenance_identity(
                            archives
                        )

    def test_archive_provenance_rejects_malformed_duplicate_and_inconsistent_fields(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        name = sorted(archives)[0]
        row = archives[name]
        malformed_values = (
            "{not-json",
            (
                '{"url":"duplicate","url":"duplicate",'
                '"archive_info":{"hash":"sha256=' + row["sha256"] + '"}}'
            ),
        )
        for direct_text in malformed_values:
            with self.subTest(direct_text=direct_text[:24]):
                with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
                    fixtures[name].direct_text = direct_text
                    with (
                        mock.patch.object(
                            public_backend.importlib.metadata,
                            "distributions",
                            return_value=list(fixtures.values()),
                        ),
                        self.assertRaisesRegex(PublicIOError, "malformed"),
                    ):
                        public_backend._science_gpu_archive_provenance_identity(
                            archives
                        )

        with _archive_fixture_environment(public_backend, archives) as (fixtures, _root):
            fixtures[name].direct_text = json.dumps(
                {
                    "archive_info": {
                        "hash": "sha256=" + "0" * 64,
                        "hashes": {"sha256": row["sha256"]},
                    },
                    "url": row["url"],
                },
                sort_keys=True,
            )
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=list(fixtures.values()),
                ),
                self.assertRaisesRegex(PublicIOError, "SHA256 differs"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

    def test_archive_provenance_requires_exact_count_and_unique_physical_distribution(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        reduced = dict(archives)
        reduced.pop(sorted(reduced)[0])
        with self.assertRaisesRegex(PublicIOError, "lock count"):
            public_backend._science_gpu_archive_provenance_identity(reduced)

        with _archive_fixture_environment(public_backend, archives) as (fixtures, root):
            name = sorted(fixtures)[0]
            same_physical_distribution = _FakeArchiveDistribution(
                name,
                archives[name],
                fixtures[name]._path,
            )
            same_path_candidates = [
                *fixtures.values(),
                same_physical_distribution,
            ]
            with mock.patch.object(
                public_backend.importlib.metadata,
                "distributions",
                return_value=same_path_candidates,
            ) as enumerate_distributions:
                receipt = public_backend._science_gpu_archive_provenance_identity(
                    archives
                )
            enumerate_distributions.assert_called_once_with(path=[str(root)])
            self.assertEqual(receipt["count"], len(archives))

            duplicate_path = root / "physically_distinct_duplicate.dist-info"
            duplicate_path.mkdir()
            distinct_distribution = _FakeArchiveDistribution(
                name,
                archives[name],
                duplicate_path,
            )
            distinct_candidates = [*fixtures.values(), distinct_distribution]
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=distinct_candidates,
                ),
                self.assertRaisesRegex(PublicIOError, "missing or duplicated"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

    def test_archive_enumeration_ignores_nested_setuptools_vendor_metadata(self) -> None:
        from compag_curation import public_backend

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            top_level = root / "packaging-26.3.dist-info"
            top_level.mkdir()
            (top_level / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: packaging\nVersion: 26.3\n",
                encoding="utf-8",
            )
            (top_level / "direct_url.json").write_text(
                '{"archive_info":{"hash":"sha256=' + "a" * 64
                + '"},"url":"https://example.invalid/packaging.whl"}',
                encoding="utf-8",
            )
            vendored = root / "setuptools/_vendor/packaging-26.0.dist-info"
            vendored.mkdir(parents=True)
            (vendored / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: packaging\nVersion: 26.0\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                public_backend,
                "_science_gpu_distribution_roots",
                return_value=(root,),
            ):
                rows = public_backend._science_gpu_archive_installation_rows(
                    ["packaging"]
                )
        self.assertEqual(len(rows["packaging"]), 1)
        self.assertEqual(rows["packaging"][0]["version"], "26.3")

    def test_archive_enumeration_rejects_unsafe_or_out_of_root_expected_metadata(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_lock_identity

        archives = dict(science_gpu_lock_identity()["archives"])
        with _archive_fixture_environment(public_backend, archives) as (fixtures, root):
            name = sorted(fixtures)[0]
            nested_path = root / "vendor/nested.dist-info"
            nested_path.mkdir(parents=True)
            nested = _FakeArchiveDistribution(name, archives[name], nested_path)
            candidates = [*fixtures.values(), nested]
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=candidates,
                ),
                self.assertRaisesRegex(PublicIOError, "outside an installation root"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

            missing_path = _FakeArchiveDistribution(
                name,
                archives[name],
                fixtures[name]._path,
            )
            del missing_path._path
            candidates = [*fixtures.values(), missing_path]
            with (
                mock.patch.object(
                    public_backend.importlib.metadata,
                    "distributions",
                    return_value=candidates,
                ),
                self.assertRaisesRegex(PublicIOError, "metadata path is invalid"),
            ):
                public_backend._science_gpu_archive_provenance_identity(archives)

    def test_archive_install_roots_are_deduplicated_and_prefix_scoped(self) -> None:
        from compag_curation import public_backend

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve(strict=True)
            root = prefix / "lib/python3.12/site-packages"
            root.mkdir(parents=True)
            with (
                mock.patch.object(public_backend.sys, "prefix", str(prefix)),
                mock.patch.object(
                    public_backend.sysconfig,
                    "get_path",
                    return_value=str(root),
                ) as get_path,
            ):
                roots = public_backend._science_gpu_distribution_roots()
        self.assertEqual(roots, (root,))
        self.assertEqual(
            [call.args for call in get_path.call_args_list],
            [("purelib",), ("platlib",)],
        )

    def test_gpu_dependency_identity_binds_compact_archive_receipt(self) -> None:
        from compag_curation import public_backend

        archive_receipt = {
            "schema": "compag-curation-science-gpu-archive-provenance/v1",
            "count": 62,
            "names": ["fixture"],
            "sha256": "a" * 64,
        }
        sam2_direct = {
            "url": "https://github.com/facebookresearch/sam2.git",
            "vcs_info": {"commit_id": "b" * 40, "vcs": "git"},
        }
        sam2_distribution = SimpleNamespace(
            read_text=lambda name: (
                json.dumps(sam2_direct)
                if name == "direct_url.json"
                else None
            )
        )
        with (
            mock.patch.object(
                public_backend,
                "science_gpu_lock_identity",
                return_value={
                    "archives": {"fixture": {}},
                    "resource": "resources/science_gpu_lock.json",
                    "sha256": "c" * 64,
                },
            ),
            mock.patch.object(
                public_backend,
                "_science_gpu_archive_provenance_identity",
                return_value=archive_receipt,
            ) as validate_archives,
            mock.patch.object(
                public_backend,
                "package_implementation_identity",
                return_value={"schema": "fixture-package/v1", "sha256": "d" * 64},
            ),
            mock.patch.object(
                public_backend,
                "EXPECTED_SCIENCE_GPU_VERSIONS",
                {"sam-2": "1.0"},
            ),
            mock.patch.object(
                public_backend.importlib.metadata,
                "version",
                return_value="1.0",
            ),
            mock.patch.object(
                public_backend.importlib.metadata,
                "distribution",
                return_value=sam2_distribution,
            ),
        ):
            identity = public_backend.gpu_dependency_identity()
        validate_archives.assert_called_once_with({"fixture": {}})
        self.assertEqual(identity["archive_provenance"], archive_receipt)
        self.assertNotIn("https://", repr(identity["archive_provenance"]))

    def test_static_identity_error_names_both_supported_environment_kinds(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError

        with (
            mock.patch.object(public_backend, "_require_supported_python"),
            mock.patch.object(
                public_backend.platform,
                "python_version",
                return_value=public_backend.SCIENCE_GPU_PYTHON,
            ),
            mock.patch.object(
                public_backend,
                "package_implementation_identity",
                side_effect=RuntimeError("fixture rejection"),
            ),
            self.assertRaisesRegex(
                PublicIOError,
                "active virtual or attested Conda/Mamba environment",
            ) as captured,
        ):
            public_backend._science_gpu_static_identity("cuda")
        self.assertIn("installation attestation failed: fixture rejection", str(captured.exception))

    def test_cuda_device_resolution_never_falls_back(self) -> None:
        from compag_curation.runtime_lock import (
            cuda_visibility_identity,
            preserved_cuda_visibility_environment,
            requested_cuda_device,
        )

        self.assertEqual(requested_cuda_device("cuda"), ("cuda:0", 0))
        self.assertEqual(requested_cuda_device("cuda:7"), ("cuda:7", 7))
        for invalid in ("cpu", "auto", "cuda:-1", "cuda:01", "cuda:gpu"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "requires device=cuda"):
                requested_cuda_device(invalid)
        with mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "3",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
            clear=True,
        ):
            visibility = cuda_visibility_identity()
            self.assertEqual(visibility["mode"], "EXPLICIT_ORDINALS")
            self.assertEqual(visibility["selector_count"], 1)
            self.assertEqual(
                visibility["selector_sha256"],
                hashlib.sha256(b"3").hexdigest(),
            )
            self.assertEqual(visibility["cuda_device_order"], "PCI_BUS_ID")
            self.assertEqual(
                preserved_cuda_visibility_environment(visibility),
                {
                    "CUDA_VISIBLE_DEVICES": "3",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                },
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            visibility = cuda_visibility_identity()
            self.assertEqual(visibility["mode"], "UNSET_ALL_VISIBLE")
            self.assertIsNone(visibility["selector_sha256"])
            self.assertEqual(preserved_cuda_visibility_environment(visibility), {})
        uuid_selector = "GPU-12345678-1234-1234-1234-123456789abc"
        uuid_visibility = cuda_visibility_identity(
            {
                "CUDA_VISIBLE_DEVICES": uuid_selector,
                "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
            }
        )
        self.assertEqual(uuid_visibility["mode"], "EXPLICIT_GPU_UUIDS")
        self.assertNotIn(uuid_selector, repr(uuid_visibility))
        self.assertEqual(uuid_visibility["cuda_device_order"], "FASTEST_FIRST")
        for invalid_visibility in ("", " 3", "3,3", "3,GPU-12345678-1234-1234-1234-123456789abc"):
            with (
                self.subTest(cuda_visible_devices=invalid_visibility),
                mock.patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": invalid_visibility},
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES|duplicate"),
            ):
                cuda_visibility_identity()
        with self.assertRaisesRegex(RuntimeError, "CUDA_DEVICE_ORDER"):
            cuda_visibility_identity({"CUDA_DEVICE_ORDER": "UNSTABLE_ORDER"})

        class MutatingEnvironment(dict[str, str]):
            def get(self, key: str, default: object = None) -> object:
                value = super().get(key, default)
                if key == "CUDA_VISIBLE_DEVICES":
                    self[key] = "4"
                return value

        snapshot_source = MutatingEnvironment(
            {
                "CUDA_VISIBLE_DEVICES": "3",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
        expected = cuda_visibility_identity(dict(snapshot_source))
        from compag_curation import runtime_lock

        with mock.patch.object(runtime_lock.os, "environ", snapshot_source):
            self.assertEqual(
                preserved_cuda_visibility_environment(expected),
                {
                    "CUDA_VISIBLE_DEVICES": "3",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                },
            )

    def test_environment_must_precede_torch_cuda_initialization(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        policy = science_gpu_runtime_policy()
        initialized = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with mock.patch.dict(public_backend.sys.modules, {"torch": initialized}, clear=False):
            with self.assertRaisesRegex(PublicIOError, "before Torch initializes"):
                public_backend._set_science_gpu_environment(policy)
        uninitialized = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        with (
            mock.patch.dict(public_backend.sys.modules, {"torch": uninitialized}, clear=False),
            mock.patch.dict(
                os.environ,
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                },
                clear=False,
            ),
        ):
            public_backend._set_science_gpu_environment(policy)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "0")

    def test_runtime_rejects_inherited_torch_tf32_cublas_override(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        with (
            mock.patch.dict(
                os.environ,
                {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"},
                clear=False,
            ),
            self.assertRaisesRegex(PublicIOError, "TF32 cuBLAS override"),
        ):
            public_backend._enforce_science_gpu_runtime(
                science_gpu_runtime_policy(),
                requested_device="cuda",
                torch_module=_FakeTorch,
                xgboost_module=_FakeXGBoost,
                cupy_module=_FakeCupy,
                sam2_extension_module=object(),
                sam2_distribution=object(),
                cv2_module=_FakeCv2,
                threadpoolctl_module=_FakeThreadpoolctl,
            )

    def test_runtime_records_device_driver_and_cuda_xgboost_build(self) -> None:
        from compag_curation import public_backend
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        with mock.patch.object(
            public_backend,
            "_sam2_cuda_extension_attestation",
            return_value=_FAKE_SAM2_ATTESTATION,
        ):
            result = public_backend._enforce_science_gpu_runtime(
                science_gpu_runtime_policy(),
                requested_device="cuda:1",
                torch_module=_FakeTorch,
                xgboost_module=_FakeXGBoost,
                cupy_module=_FakeCupy,
                sam2_extension_module=object(),
                sam2_distribution=object(),
                cv2_module=_FakeCv2,
                threadpoolctl_module=_FakeThreadpoolctl,
            )
        self.assertEqual(result["requested_device"], "cuda:1")
        self.assertEqual(result["resolved_device"], "cuda:1")
        self.assertEqual(result["cuda"]["device_index"], 1)
        self.assertEqual(result["cuda"]["device_name"], "Fixture GPU 1")
        self.assertEqual(result["cuda"]["compute_capability"], [8, 9])
        self.assertEqual(result["cuda"]["torch_cuda_runtime"], "13.2")
        self.assertEqual(result["cuda"]["driver_api_version"], 13020)
        self.assertEqual(result["xgboost_build"]["USE_CUDA"], True)
        self.assertNotIn("libxgboost", result["xgboost_build"])
        self.assertEqual(_FakeCuda.synchronized, [1])

    def test_runtime_drops_cuda_probe_from_a_retained_torch_import_frame(self) -> None:
        from compag_curation import public_backend
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        retained_frames: list[object] = []

        def retain_caller_frame(enabled: bool) -> None:
            _FakeTorch.deterministic = enabled
            retained_frames.append(public_backend.sys._getframe(1))

        with (
            mock.patch.object(
                _FakeTorch,
                "use_deterministic_algorithms",
                side_effect=retain_caller_frame,
            ),
            mock.patch.object(
                public_backend,
                "_sam2_cuda_extension_attestation",
                return_value=_FAKE_SAM2_ATTESTATION,
            ),
        ):
            public_backend._enforce_science_gpu_runtime(
                science_gpu_runtime_policy(),
                requested_device="cuda",
                torch_module=_FakeTorch,
                xgboost_module=_FakeXGBoost,
                cupy_module=_FakeCupy,
                sam2_extension_module=object(),
                sam2_distribution=object(),
                cv2_module=_FakeCv2,
                threadpoolctl_module=_FakeThreadpoolctl,
            )
        self.assertEqual(len(retained_frames), 1)
        self.assertNotIn("probe", retained_frames[0].f_locals)

    def test_runtime_rejects_missing_cuda_and_cpu_xgboost_build(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        _FakeCuda.available = False
        with self.assertRaisesRegex(PublicIOError, "CPU fallback is disabled"):
            public_backend._enforce_science_gpu_runtime(
                science_gpu_runtime_policy(),
                requested_device="cuda",
                torch_module=_FakeTorch,
                xgboost_module=_FakeXGBoost,
                cupy_module=_FakeCupy,
                sam2_extension_module=object(),
                sam2_distribution=object(),
                cv2_module=_FakeCv2,
                threadpoolctl_module=_FakeThreadpoolctl,
            )
        with self.assertRaisesRegex(PublicIOError, "USE_CUDA=true"):
            public_backend._xgboost_cuda_build_identity(
                SimpleNamespace(build_info=lambda: {"USE_CUDA": False})
            )
        _FakeCuda.available = True
        _FakeTorch.autocast_enabled = True
        with self.assertRaisesRegex(PublicIOError, "autocast must be disabled"):
            public_backend._enforce_science_gpu_runtime(
                science_gpu_runtime_policy(),
                requested_device="cuda",
                torch_module=_FakeTorch,
                xgboost_module=_FakeXGBoost,
                cupy_module=_FakeCupy,
                sam2_extension_module=object(),
                sam2_distribution=object(),
                cv2_module=_FakeCv2,
                threadpoolctl_module=_FakeThreadpoolctl,
            )

    def test_require_science_dependencies_dispatches_cuda_requests(self) -> None:
        from compag_curation import public_backend

        expected = {"schema": "fixture-science-gpu/v1"}
        with mock.patch.object(public_backend, "require_science_gpu_dependencies", return_value=expected) as required:
            self.assertIs(public_backend.require_science_dependencies("cuda:1"), expected)
        required.assert_called_once_with("cuda:1")

    def test_gpu_revalidation_accepts_existing_context_only_with_exact_receipt(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        identity = {
            "schema": "fixture-static-gpu-identity/v1",
            "versions": {"fixture": "1"},
            "package_implementation": {"sha256": "a" * 64},
        }
        installed = {
            "sha256": "a" * 64,
            "installation": "SYS_PREFIX_DISTRIBUTION_NON_EDITABLE_ISOLATED_NO_BYTECODE",
        }

        def imported(name: str) -> object:
            if name == "torch":
                return _FakeTorch
            if name == "xgboost":
                return _FakeXGBoost
            if name == "cupy":
                return _FakeCupy
            if name == "cv2":
                return _FakeCv2
            if name == "threadpoolctl":
                return _FakeThreadpoolctl
            return object()

        with mock.patch.object(
            public_backend,
            "_sam2_cuda_extension_attestation",
            return_value=_FAKE_SAM2_ATTESTATION,
        ):
            enforced = public_backend._enforce_science_gpu_runtime(
                science_gpu_runtime_policy(),
                requested_device="cuda",
                torch_module=_FakeTorch,
                xgboost_module=_FakeXGBoost,
                cupy_module=_FakeCupy,
                sam2_extension_module=object(),
                sam2_distribution=object(),
                cv2_module=_FakeCv2,
                threadpoolctl_module=_FakeThreadpoolctl,
            )
        receipt = public_backend._science_gpu_dependency_record(
            identity,
            installed,
            list(public_backend.SCIENCE_GPU_IMPORTS),
            enforced,
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                },
                clear=False,
            ),
            mock.patch.object(
                public_backend,
                "_science_gpu_static_identity",
                return_value=(dict(identity), dict(installed)),
            ),
            mock.patch.object(
                public_backend.importlib,
                "import_module",
                side_effect=imported,
            ),
            mock.patch.object(
                public_backend.importlib.metadata,
                "distribution",
                return_value=object(),
            ),
            mock.patch.object(
                public_backend,
                "_sam2_cuda_extension_attestation",
                return_value=_FAKE_SAM2_ATTESTATION,
            ),
            mock.patch.object(
                public_backend,
                "_set_science_gpu_environment",
                side_effect=AssertionError("pre-context setup must not be repeated"),
            ) as pre_context,
        ):
            self.assertEqual(
                public_backend.revalidate_science_gpu_dependencies("cuda", receipt),
                receipt,
            )
            pre_context.assert_not_called()
            _FakeCudnn.allow_tf32 = True
            with self.assertRaisesRegex(PublicIOError, "TF32 disablement"):
                public_backend.revalidate_science_gpu_dependencies("cuda", receipt)
            _FakeCudnn.allow_tf32 = False
            _FakeThreadpoolctl.threads = 2
            with self.assertRaisesRegex(PublicIOError, "thread-pool limit"):
                public_backend.revalidate_science_gpu_dependencies("cuda", receipt)
            _FakeThreadpoolctl.threads = 1
            with (
                mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}, clear=False),
                self.assertRaisesRegex(PublicIOError, "environment changed"),
            ):
                public_backend.revalidate_science_gpu_dependencies("cuda", receipt)

    def test_gpu_revalidation_rejects_archive_provenance_drift(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_gpu_runtime_policy

        installed = {"installation": "fixture-installation"}
        enforced = {
            "policy": science_gpu_runtime_policy(),
            "requested_device": "cuda",
            "resolved_device": "cuda:0",
            "cuda": {"device_index": 0},
            "xgboost_build": {"USE_CUDA": True},
            "sam2_cuda_extension": _FAKE_SAM2_ATTESTATION,
        }
        captured_identity = {
            "schema": "fixture/v1",
            "versions": {"fixture": "1"},
            "archive_provenance": {
                "schema": "compag-curation-science-gpu-archive-provenance/v1",
                "count": 62,
                "names": ["fixture"],
                "sha256": "a" * 64,
            },
        }
        receipt = public_backend._science_gpu_dependency_record(
            captured_identity,
            installed,
            list(public_backend.SCIENCE_GPU_IMPORTS),
            enforced,
        )
        drifted_identity = {
            **captured_identity,
            "archive_provenance": {
                **captured_identity["archive_provenance"],
                "sha256": "b" * 64,
            },
        }
        with (
            mock.patch.dict(
                os.environ,
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                },
                clear=False,
            ),
            mock.patch.object(
                public_backend,
                "_science_gpu_static_identity",
                return_value=(drifted_identity, installed),
            ),
            mock.patch.object(
                public_backend,
                "_enforce_science_gpu_runtime",
                return_value=enforced,
            ),
            mock.patch.object(
                public_backend.importlib,
                "import_module",
                return_value=object(),
            ),
            mock.patch.object(
                public_backend.importlib.metadata,
                "distribution",
                return_value=object(),
            ),
        ):
            with self.assertRaisesRegex(PublicIOError, "identity changed"):
                public_backend.revalidate_science_gpu_dependencies("cuda", receipt)

    def test_historical_cpu_preflight_is_rejected_before_runtime_imports(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError

        with (
            mock.patch.object(public_backend, "_require_supported_python") as runtime,
            mock.patch.object(public_backend.importlib, "import_module") as imported,
        ):
            with self.assertRaisesRegex(PublicIOError, "historical verification-only"):
                public_backend.require_science_dependencies("cpu")
        runtime.assert_not_called()
        imported.assert_not_called()

    def test_historical_balanced_inference_is_never_executable(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError

        with (
            mock.patch.object(public_backend, "verify_model_bundle") as verified,
            mock.patch.object(public_backend, "require_science_dependencies") as required,
            self.assertRaisesRegex(PublicIOError, "historical verification-only"),
        ):
            public_backend.infer_from_bundle(
                Path("images"),
                Path("historical-bundle"),
                Path("output"),
                device="cpu",
            )
        verified.assert_not_called()
        required.assert_not_called()

    def test_sam2_compiled_extension_is_hash_bound_and_cuda_probed(self) -> None:
        from compag_curation import public_backend

        class Scalar:
            def item(self) -> int:
                return 0

        class Tensor:
            def __init__(self, dtype: object) -> None:
                self.shape = (1, 1, 4, 4)
                self.dtype = dtype
                self.is_cuda = True
                self.device = SimpleNamespace(type="cuda", index=0)

        uint8 = object()
        int32 = object()

        class Torch:
            cuda = _FakeCuda

            @staticmethod
            def zeros(shape, *, dtype, device):
                self.assertEqual(shape, (1, 1, 4, 4))
                self.assertIs(dtype, uint8)
                self.assertEqual(device, "cuda:0")
                return Tensor(uint8)

            @staticmethod
            def count_nonzero(_tensor):
                return Scalar()

        Torch.uint8 = uint8
        Torch.int32 = int32

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extension = root / "sam2/_C.so"
            extension.parent.mkdir()
            extension.write_bytes(b"fixture-sam2-cuda-extension\n")

            class Distribution:
                version = "1.0"
                files = ("sam2/_C.so",)

                @staticmethod
                def locate_file(_entry: object) -> Path:
                    return extension

            extension_module = SimpleNamespace(
                __file__=str(extension),
                get_connected_componnets=lambda value: [Tensor(int32), Tensor(int32)],
            )
            result = public_backend._sam2_cuda_extension_attestation(
                torch_module=Torch,
                extension_module=extension_module,
                distribution=Distribution(),
                resolved_device="cuda:0",
                device_index=0,
            )
        self.assertEqual(result["distribution_path"], "sam2/_C.so")
        self.assertEqual(result["size_bytes"], len(b"fixture-sam2-cuda-extension\n"))
        self.assertRegex(str(result["sha256"]), r"^[0-9a-f]{64}$")
        self.assertEqual(result["probe"]["outputs"][0]["dtype"], "int32")
        self.assertIs(result["probe"]["cpu_fallback"], False)

    def test_real_sam2_cuda_extension_smoke_when_runtime_is_available(self) -> None:
        from compag_curation import public_backend

        try:
            import importlib.metadata
            import torch
            import sam2._C as extension
        except (ImportError, OSError) as exc:
            self.skipTest(f"locked SAM2 GPU runtime is unavailable: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        result = public_backend._sam2_cuda_extension_attestation(
            torch_module=torch,
            extension_module=extension,
            distribution=importlib.metadata.distribution("sam-2"),
            resolved_device="cuda:0",
            device_index=0,
        )
        self.assertEqual(result["status"] if "status" in result else "PASS", "PASS")
        self.assertIs(result["probe"]["cpu_fallback"], False)

    def test_parent_gpu_allocator_caches_are_released_before_child_inference(self) -> None:
        from compag_curation import public_backend

        device_pool = _FakePool(128, 1024)
        pinned_pool = _FakePool(0, 0, free_blocks=4)
        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(Device=_FakeCupyDevice),
            get_default_memory_pool=lambda: device_pool,
            get_default_pinned_memory_pool=lambda: pinned_pool,
        )

        def imported(name: str) -> object:
            if name == "torch":
                return _FakeTorch
            if name == "cupy":
                return fake_cupy
            raise AssertionError(name)

        with (
            mock.patch.object(
                public_backend.importlib,
                "import_module",
                side_effect=imported,
            ),
            mock.patch.object(public_backend.gc, "collect", side_effect=(3, 2)),
        ):
            result = public_backend.release_science_gpu_allocator_caches("cuda")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["schema"],
            "compag-curation-science-gpu-parent-allocator-cache-release/v1",
        )
        self.assertEqual(
            result["validation_scope"],
            "OBSERVED_TORCH_AND_CUPY_ALLOCATOR_COUNTERS_ONLY",
        )
        self.assertEqual(
            result["native_cuda_context_xgboost_vram_zero_proof"],
            "NOT_CLAIMED",
        )
        self.assertNotIn("object_lifetime_boundary", result)
        self.assertIs(result["cpu_fallback"], False)
        self.assertEqual(result["gc_collected"], 5)
        self.assertEqual(result["torch_before"]["reserved_bytes"], 512)
        self.assertEqual(result["torch_after"]["reserved_bytes"], 0)
        self.assertEqual(result["cupy_before"]["total_bytes"], 1024)
        self.assertEqual(result["cupy_after"]["total_bytes"], 0)
        self.assertEqual(_FakeTorchInternal.clear_cublas_workspaces_calls, 1)
        self.assertEqual(
            result["actions"],
            [
                "CUDA_SYNCHRONIZE",
                "CUPY_DEVICE_SYNCHRONIZE",
                "TORCH_CUDA_CLEAR_CUBLAS_WORKSPACES",
                "POST_CUBLAS_WORKSPACE_RELEASE_SYNCHRONIZE",
                "PYTHON_GC_COLLECT",
                "CUPY_DEFAULT_POOL_FREE_ALL_BLOCKS",
                "CUPY_PINNED_POOL_FREE_ALL_BLOCKS",
                "TORCH_CUDA_EMPTY_CACHE",
                "POST_RELEASE_SYNCHRONIZE",
            ],
        )
        self.assertEqual(_FakeCuda.synchronized, [0, 0, 0])
        self.assertEqual(_FakeCupyDevice.synchronized, [0, 0])

    def test_parent_gpu_allocator_release_fails_closed_without_cublas_cleanup_api(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError

        device_pool = _FakePool(0, 0)
        pinned_pool = _FakePool(0, 0)
        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(Device=_FakeCupyDevice),
            get_default_memory_pool=lambda: device_pool,
            get_default_pinned_memory_pool=lambda: pinned_pool,
        )

        def imported(name: str) -> object:
            if name == "torch":
                return _FakeTorch
            if name == "cupy":
                return fake_cupy
            raise AssertionError(name)

        with (
            mock.patch.object(_FakeTorch, "_C", SimpleNamespace()),
            mock.patch.object(
                public_backend.importlib,
                "import_module",
                side_effect=imported,
            ),
            self.assertRaisesRegex(PublicIOError, "cuBLAS workspace release API"),
        ):
            public_backend.release_science_gpu_allocator_caches("cuda")

    def test_stage60_releases_parent_gpu_allocator_caches_before_nonzero_mapped_spawn(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.public_io import (
            compact_json_sha256,
            owned_manifest_contract,
            write_new_bytes,
            write_new_json,
        )
        from compag_curation.runtime_lock import cuda_visibility_identity

        class ExpectedSpawnStop(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            bundle = root / "bundle"
            output = root / "stage60"
            images.mkdir()
            bundle.mkdir()
            output.mkdir()
            write_new_json(bundle / "bundle.json", {"schema": "fixture-bundle/v1"})
            config = SimpleNamespace(device="cuda", inference_images=images)
            events: list[str] = []
            release_record = {
                "schema": "compag-curation-science-gpu-parent-allocator-cache-release/v1",
                "status": "PASS",
                "cpu_fallback": False,
            }

            def release(device: str) -> dict[str, object]:
                self.assertEqual(device, "cuda")
                events.append("release")
                return release_record

            def spawn(
                _argv: object,
                *,
                cwd: Path,
                env: dict[str, str],
                acceptable_exit_codes: frozenset[int],
                capture_output: bool,
            ) -> object:
                self.assertEqual(events, ["release"])
                self.assertEqual(cwd, output)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
                for key in (
                    "CUDA_CACHE_PATH",
                    "CUPY_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                ):
                    self.assertEqual(
                        env[key],
                        str(output / "fresh_process_runtime"),
                    )
                self.assertEqual(acceptable_exit_codes, frozenset({0}))
                self.assertTrue(capture_output)
                self.assertTrue(
                    (
                        output
                        / "fresh_process_runtime/GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json"
                    ).is_file()
                )
                cache = output / "fresh_process_runtime/9/0"
                cache.mkdir(parents=True, mode=0o700)
                write_new_bytes(
                    cache / "c25eed8a829ad6",
                    b"compiled cache containing /home/private\n",
                )
                events.append("spawn")
                raise ExpectedSpawnStop

            with mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "3"},
                clear=False,
            ):
                visibility = cuda_visibility_identity()
            dependencies: dict[str, object] = {
                "schema": "compag-curation-science-gpu-dependencies/v2",
                "requested_device": "cuda",
                "cuda_visibility": visibility,
            }
            dependencies["sha256"] = compact_json_sha256(dependencies)
            with (
                mock.patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "3"},
                    clear=False,
                ),
                mock.patch.object(
                    public_pipeline,
                    "_input_images_identity",
                    return_value=[
                        {
                            "filename": "fixture.ppm",
                            "sha256": "a" * 64,
                            "size_bytes": 1,
                            "width": 64,
                            "height": 64,
                            "exif_orientation": None,
                        }
                    ],
                ),
                mock.patch(
                    "compag_curation.public_backend.release_science_gpu_allocator_caches",
                    side_effect=release,
                ),
                mock.patch(
                    "compag_curation.domain.inference.LocalCommandRunner.run",
                    side_effect=spawn,
                ),
            ):
                with self.assertRaises(ExpectedSpawnStop):
                    public_pipeline._fresh_canonical_inference(
                        config,
                        bundle,
                        output,
                        dependencies,
                    )
            self.assertEqual(events, ["release", "spawn"])
            files, directories = owned_manifest_contract(
                output / "fresh_process_runtime"
            )
            self.assertEqual(
                [row["path"] for row in files],
                [
                    "ENVIRONMENT_POLICY.json",
                    "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
                ],
            )
            self.assertEqual(
                directories,
                [{"path": ".", "mode_octal": "0700"}],
            )

    def test_nested_stage60_child_environment_requires_exact_parent_receipt(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.public_io import PublicIOError, compact_json_sha256, write_new_json

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            dependencies: dict[str, object] = {
                "schema": "compag-curation-science-gpu-dependencies/v2",
                "requested_device": "cuda",
            }
            dependencies["sha256"] = compact_json_sha256(dependencies)
            run_id = "11111111-1111-4111-8111-111111111111"
            write_new_json(
                output / "FINAL_STATUS.json",
                {"schema": "fixture-status/v1", "run_id": run_id},
            )
            write_new_json(
                output / "environment.json",
                {
                    "schema": "compag-curation-inference-environment/v1",
                    "run_id": run_id,
                    "dependencies": dependencies,
                },
            )
            validated = public_pipeline._validated_fresh_inference_environment(
                output,
                dependencies,
            )
            self.assertEqual(validated["dependencies"], dependencies)
            drifted = dict(dependencies)
            drifted["sha256"] = "f" * 64
            with self.assertRaisesRegex(PublicIOError, "differs from run preflight"):
                public_pipeline._validated_fresh_inference_environment(
                    output,
                    drifted,
                )

    def test_post_producer_preflight_drift_never_publishes_and_retry_recomputes(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.public_io import PublicIOError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "stages/50_train_bundle"
            attempts = iter((root / "attempt-1", root / "attempt-2"))
            produced: list[Path] = []
            guard_calls = 0
            drift = True

            def begin(_state: object, _name: str) -> object:
                workspace = next(attempts)
                workspace.mkdir()
                return SimpleNamespace(root=workspace)

            def guard() -> None:
                nonlocal guard_calls
                guard_calls += 1
                if drift and guard_calls == 2:
                    raise PublicIOError("fixture dependency drift")

            def producer(workspace: Path) -> None:
                produced.append(workspace)
                (workspace / "result.json").write_text("{}\n", encoding="ascii")

            def publish(_state: object, _workspace: object, _required: object) -> dict[str, object]:
                target.mkdir(parents=True)
                return {"status": "PASS", "required": ["result.json"]}

            with (
                mock.patch.object(public_pipeline, "completed_stage", return_value=None),
                mock.patch.object(public_pipeline, "begin_stage", side_effect=begin),
                mock.patch.object(public_pipeline, "publish_stage", side_effect=publish) as published,
                mock.patch.object(public_pipeline, "record_stage_failure") as failed,
            ):
                with self.assertRaisesRegex(PublicIOError, "dependency drift"):
                    public_pipeline._produce_stage(
                        object(),
                        "50_train_bundle",
                        ("result.json",),
                        producer,
                        dependency_guard=guard,
                    )
                self.assertFalse(target.exists())
                published.assert_not_called()
                failed.assert_called_once()
                drift = False
                result = public_pipeline._produce_stage(
                    object(),
                    "50_train_bundle",
                    ("result.json",),
                    producer,
                    dependency_guard=guard,
                )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(target.is_dir())
            self.assertEqual(produced, [root / "attempt-1", root / "attempt-2"])

    def test_completed_gpu_stage_is_accepted_only_after_guard(self) -> None:
        from compag_curation import public_pipeline

        existing = {"status": "PASS", "required": ["result.json"]}
        guard = mock.Mock()
        with (
            mock.patch.object(public_pipeline, "completed_stage", return_value=existing),
            mock.patch.object(public_pipeline, "begin_stage") as begin,
        ):
            result = public_pipeline._produce_stage(
                object(),
                "50_train_bundle",
                ("result.json",),
                mock.Mock(),
                dependency_guard=guard,
            )
        self.assertIs(result, existing)
        guard.assert_called_once_with()
        begin.assert_not_called()

    def test_v13_execution_rejects_historical_profiles_before_preflight(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.public_io import PublicIOError

        historical = SimpleNamespace(
            profile="canonical-xgb-recall-cpu-v1",
            device="cpu",
        )
        with (
            mock.patch.object(public_pipeline, "load_public_config", return_value=historical),
            mock.patch.object(public_pipeline, "_preflight") as preflight,
        ):
            with self.assertRaisesRegex(PublicIOError, "historical verification-only"):
                public_pipeline.run_project(Path("unused.toml"), output=Path("unused"))
            with self.assertRaisesRegex(PublicIOError, "historical verification-only"):
                public_pipeline.validate_project(Path("unused.toml"))
        preflight.assert_not_called()

    def test_inference_verifies_bundle_before_gpu_only_gate_and_science_preflight(self) -> None:
        from compag_curation import model_bundle, public_backend, public_pipeline
        from compag_curation.public_io import PublicIOError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            bundle = root / "bundle"
            images.mkdir()
            bundle.mkdir()
            historical = SimpleNamespace(
                schema="historical-bundle/v1",
                profile="balanced-xgb-cpu-v1",
            )
            events: list[str] = []

            def verify(_path: Path) -> object:
                events.append("verify")
                return historical

            with (
                mock.patch.object(model_bundle, "verify_model_bundle", side_effect=verify),
                mock.patch.object(public_backend, "require_science_dependencies") as require,
            ):
                with self.assertRaisesRegex(PublicIOError, "verification-only"):
                    public_pipeline.infer_bundle(
                        images,
                        bundle,
                        root / "output",
                    )
            self.assertEqual(events, ["verify"])
            require.assert_not_called()

    def test_science_cpu_doctor_is_structured_unsupported_without_import(self) -> None:
        from compag_curation import public_backend, public_pipeline

        with mock.patch.object(public_backend, "require_science_dependencies") as require:
            result, code = public_pipeline.doctor("science-cpu")
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertIn("historical verification-only", result["reason"])
        require.assert_not_called()

    def test_standalone_gpu_inference_revalidates_before_sealing_and_rejects_drift(self) -> None:
        from compag_curation import model_bundle, public_backend, public_pipeline
        from compag_curation.public_config import CANONICAL_GPU_PROFILE
        from compag_curation.public_io import (
            PublicIOError,
            compact_json_sha256,
            write_new_json,
        )

        def execute(*, drift: bool) -> tuple[Path, list[str]]:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
            images = root / "images"
            bundle = root / "bundle"
            output = root / ("drift-output" if drift else "success-output")
            images.mkdir()
            bundle.mkdir()
            events: list[str] = []
            dependencies: dict[str, object] = {
                "schema": "compag-curation-science-gpu-dependencies/v2",
                "requested_device": "cuda",
            }
            dependencies["sha256"] = compact_json_sha256(dependencies)
            verified = SimpleNamespace(
                schema=model_bundle.BUNDLE_SCHEMA_V2,
                profile=CANONICAL_GPU_PROFILE,
                bundle_sha256="a" * 64,
                tree_identity=(("bundle.json", "b" * 64),),
                normalized_config={"profile": CANONICAL_GPU_PROFILE},
                manifest={"feature_order_sha256": "c" * 64},
                compatibility={"classifier_format": "XGBOOST_UBJ"},
            )
            image_identity = [{
                "filename": "fixture.ppm",
                "sha256": "d" * 64,
                "size_bytes": 1,
                "width": 64,
                "height": 64,
                "exif_orientation": None,
                "mode_octal": "0644",
                "uid": os.getuid(),
                "gid": os.getgid(),
                "nlink": 1,
                "mtime_ns": 1,
                "ctime_ns": 1,
            }]

            def verify(_path: Path) -> object:
                events.append("verify")
                return verified

            def require(device: str) -> dict[str, object]:
                self.assertEqual(device, "cuda")
                self.assertEqual(events[0], "verify")
                events.append("preflight")
                return dependencies

            def infer(
                _images: Path,
                _bundle: Path,
                staging: Path,
                *,
                device: str,
            ) -> dict[str, object]:
                self.assertEqual(device, "cuda")
                result = {"schema": "fixture-inference/v1", "status": "PASS"}
                write_new_json(staging / "inference_result.json", result)
                events.append("inference")
                return result

            def revalidate(
                device: str,
                captured: object,
            ) -> dict[str, object]:
                self.assertEqual(device, "cuda")
                self.assertEqual(captured, dependencies)
                staging = next(root.glob(f".{output.name}.inference.*"))
                self.assertTrue((staging / "inference_result.json").is_file())
                self.assertFalse((staging / "FINAL_STATUS.json").exists())
                events.append("revalidate")
                if drift:
                    return {**dependencies, "drift": True}
                return dependencies

            patches = (
                mock.patch.object(model_bundle, "verify_model_bundle", side_effect=verify),
                mock.patch.object(public_backend, "require_science_dependencies", side_effect=require),
                mock.patch.object(public_backend, "revalidate_science_gpu_dependencies", side_effect=revalidate),
                mock.patch.object(public_pipeline, "_input_images_identity", return_value=image_identity),
                mock.patch(
                    "compag_curation.canonical.service.infer_canonical_bundle",
                    side_effect=infer,
                ),
                mock.patch.object(
                    public_pipeline,
                    "_validated_canonical_inference_result",
                    side_effect=lambda value, **_kwargs: dict(value),
                ),
                mock.patch.object(public_pipeline, "_validated_canonical_predictions"),
                mock.patch.object(public_pipeline, "_validated_canonical_raw_feature_archive"),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                if drift:
                    with self.assertRaisesRegex(PublicIOError, "changed before publication"):
                        public_pipeline.infer_bundle(images, bundle, output, device="cuda")
                else:
                    result, code = public_pipeline.infer_bundle(
                        images,
                        bundle,
                        output,
                        device="cuda",
                    )
                    self.assertEqual(code, 0)
                    self.assertEqual(result["status"], "PASS")
            return output, events

        success_output, success_events = execute(drift=False)
        self.assertTrue((success_output / "FINAL_STATUS.json").is_file())
        self.assertEqual(success_events.count("revalidate"), 1)
        drift_output, drift_events = execute(drift=True)
        self.assertFalse(drift_output.exists())
        drift_staging = next(drift_output.parent.glob(f".{drift_output.name}.inference.*"))
        self.assertFalse((drift_staging / "OUTPUT_MANIFEST.json").exists())
        self.assertEqual(drift_events.count("revalidate"), 1)


if __name__ == "__main__":
    unittest.main()
