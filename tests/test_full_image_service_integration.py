from __future__ import annotations

import csv
import io
import json
import stat
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from compag_curation.canonical import service
from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalRawFeature,
)
from compag_curation.canonical.full_image import (
    FULL_IMAGE_COMPONENT_POLICY,
    FULL_IMAGE_DURABLE_MASK_ENCODING,
    FullImageBackend,
    FullImageProposal,
    select_full_image_proposals,
)
from compag_curation.canonical.preprocessing import (
    CanonicalPreparedImage,
    CanonicalTile,
    full_image_unit_name,
)
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH,
)
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_io import PublicIOError, bounded_csv_field_limit
from tests.test_canonical_service import _FakeEmbedder, _config, _write_ppm


def _full_image_config(root: Path) -> SimpleNamespace:
    config = _config(root)
    config.profile = FULL_IMAGE_GPU_PROFILE
    config.device = "cuda"
    config.spatial_mode = "full-image-multiscale"
    config.max_proposals_per_image = 1000
    config.execution_points_per_batch = 8
    return config


def _full_prepared(image_name: str, pixels: object) -> CanonicalPreparedImage:
    import numpy as np

    image = np.asarray(pixels, dtype=np.uint8)
    height, width = (int(value) for value in image.shape[:2])
    inverse = np.eye(3, dtype=np.float32)
    unit = CanonicalTile(
        name=full_image_unit_name(image_name),
        x=0,
        y=0,
        crop_width=width,
        crop_height=height,
        pixels=image,
    )
    return CanonicalPreparedImage(
        warped_bgr=image,
        inverse_warp=inverse,
        warp_mode="identity_fallback",
        row_lines=(100, 250, 400, 550),
        column_lines=(225, 450, 675),
        tiles=(unit,),
        evidence={
            "schema": "compag-curation-full-image-preprocessing/v1",
            "spatial_mode": "full-image-multiscale",
            "warp": {
                "mode": "identity_fallback",
                "inverse_matrix": inverse.tolist(),
            },
            "grid": {},
            "analysis_units": {
                "kind": "full_warped_frame",
                "count": 1,
                "external_tiling": False,
                "format": "png",
                "lossless": True,
                "width": width,
                "height": height,
            },
        },
    )


def _png_bytes(pixels_bgr: object) -> bytes:
    import numpy as np
    from PIL import Image

    bgr = np.asarray(pixels_bgr, dtype=np.uint8)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _rectangle_rle(
    canvas_height: int,
    canvas_width: int,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, object]:
    counts: list[int] = []
    cursor = 0
    for canvas_x in range(x, x + width):
        start = canvas_x * canvas_height + y
        counts.extend((start - cursor, height))
        cursor = start + height
    counts.append(canvas_height * canvas_width - cursor)
    return {"size": [canvas_height, canvas_width], "counts": counts}


class _FullImageRLEGenerator:
    def __init__(
        self,
        canvas_shape: tuple[int, int],
        bbox: tuple[int, int, int, int],
    ) -> None:
        self.canvas_shape = canvas_shape
        self.bbox = bbox
        self.calls = 0

    def generate(self, image: object) -> list[dict[str, object]]:
        import numpy as np

        array = np.asarray(image)
        if tuple(array.shape[:2]) != self.canvas_shape:
            raise AssertionError("full-image AMG received the wrong canvas")
        if tuple(int(value) for value in array[0, 0]) != (255, 220, 0):
            raise AssertionError("full-image AMG input was not converted BGR -> RGB")
        self.calls += 1
        x, y, width, height = self.bbox
        return [
            {
                "segmentation": _rectangle_rle(
                    *self.canvas_shape,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                ),
                "predicted_iou": 0.91,
                "stability_score": 0.93,
            }
        ]


class _FakeCV2(ModuleType):
    COLOR_BGR2RGB = 1
    RETR_EXTERNAL = 2
    CHAIN_APPROX_SIMPLE = 3
    LINE_8 = 4
    FONT_HERSHEY_SIMPLEX = 5
    IMWRITE_PNG_COMPRESSION = 6

    def __init__(self) -> None:
        super().__init__("cv2")
        self.contour_shapes: list[tuple[int, int]] = []
        self.contour_offsets: list[tuple[int, int]] = []

    def cvtColor(self, image: object, code: int) -> object:
        import numpy as np

        if code != self.COLOR_BGR2RGB:
            raise AssertionError("unexpected fake cv2 colour conversion")
        return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])

    def findContours(
        self,
        mask: object,
        _mode: int,
        _method: int,
    ) -> tuple[list[object], None]:
        import numpy as np

        array = np.asarray(mask, dtype=np.uint8)
        self.contour_shapes.append(tuple(int(value) for value in array.shape))
        contour = np.asarray([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]])
        return [contour], None

    def drawContours(self, image: object, *args: object, **kwargs: object) -> object:
        self.contour_offsets.append(tuple(kwargs.get("offset", (0, 0))))
        return image

    def getTextSize(self, text: str, *_args: object) -> tuple[tuple[int, int], int]:
        return (max(1, len(text) * 6), 8), 2

    def putText(self, image: object, *_args: object, **_kwargs: object) -> object:
        return image

    def imencode(
        self,
        extension: str,
        image: object,
        _options: object,
    ) -> tuple[bool, object]:
        import numpy as np

        if extension != ".png":
            raise AssertionError("full-image overlay was not encoded as PNG")
        return True, np.frombuffer(_png_bytes(image), dtype=np.uint8)


def _fake_gate_core() -> ModuleType:
    import numpy as np

    module = ModuleType("compag_curation.proposals.sam2_pipeline.gate_core")

    def mask_to_poly_and_bbox_in_original(
        mask: object,
        local_to_original: object,
        _source_shape: object,
    ) -> tuple[list[int], tuple[int, int, int, int]]:
        binary = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.nonzero(binary)
        if not len(xs):
            raise AssertionError("full-image geometry received an empty mask crop")
        transform = np.asarray(local_to_original)
        origin_x = int(round(float(transform[0, 2])))
        origin_y = int(round(float(transform[1, 2])))
        x = origin_x + int(xs.min())
        y = origin_y + int(ys.min())
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        return (
            [x, y, x + width, y, x + width, y + height, x, y + height],
            (x, y, width, height),
        )

    module.mask_to_poly_and_bbox_in_original = mask_to_poly_and_bbox_in_original
    return module


class FullImageServiceIntegrationTests(unittest.TestCase):
    def test_bounded_csv_limit_serializes_threads_and_restores_after_error(
        self,
    ) -> None:
        original_limit = csv.field_size_limit()
        first_entered = threading.Event()
        allow_first_exit = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        failures: list[BaseException] = []
        observed: list[int] = []

        def first_reader() -> None:
            try:
                with bounded_csv_field_limit(1024 * 1024):
                    observed.append(csv.field_size_limit())
                    first_entered.set()
                    if not allow_first_exit.wait(2.0):
                        raise AssertionError("timed out waiting to release first reader")
            except BaseException as exc:  # pragma: no cover - diagnostic capture
                failures.append(exc)

        def second_reader() -> None:
            try:
                if not first_entered.wait(2.0):
                    raise AssertionError("first reader did not acquire the CSV guard")
                second_started.set()
                with bounded_csv_field_limit(2 * 1024 * 1024):
                    observed.append(csv.field_size_limit())
                    second_entered.set()
            except BaseException as exc:  # pragma: no cover - diagnostic capture
                failures.append(exc)

        first = threading.Thread(target=first_reader)
        second = threading.Thread(target=second_reader)
        first.start()
        second.start()
        self.assertTrue(first_entered.wait(2.0))
        self.assertTrue(second_started.wait(2.0))
        self.assertFalse(second_entered.wait(0.05))
        allow_first_exit.set()
        first.join(2.0)
        second.join(2.0)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(observed, [1024 * 1024, 2 * 1024 * 1024])
        self.assertEqual(csv.field_size_limit(), original_limit)

        with self.assertRaisesRegex(RuntimeError, "forced reader failure"):
            with bounded_csv_field_limit():
                raise RuntimeError("forced reader failure")
        self.assertEqual(csv.field_size_limit(), original_limit)

    def test_full_image_csv_readers_accept_bounded_compact_masks_above_python_default(
        self,
    ) -> None:
        from compag_curation.public_pipeline import _proposal_rows

        # Regression for the first real full-image E2E mask that exceeded
        # Python's platform-default 128-KiB CSV field limit.
        large_mask = "A" * 1_745_468
        original_limit = csv.field_size_limit()
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=("mask",), lineterminator="\n")
        writer.writeheader()
        writer.writerow({"mask": large_mask})
        payload = buffer.getvalue().encode("utf-8")

        parsed = service._parse_csv_exact(payload, "large-mask.csv", ("mask",))
        self.assertEqual(parsed, [{"mask": large_mask}])
        self.assertEqual(csv.field_size_limit(), original_limit)

        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "proposals.csv"
            row = {name: "x" for name in service.CANONICAL_PROPOSAL_COLUMNS}
            row["mask_packbits_base64"] = large_mask
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=service.CANONICAL_PROPOSAL_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(row)
            self.assertEqual(
                _proposal_rows(path, canonical=True)[0]["mask_packbits_base64"],
                large_mask,
            )
        self.assertEqual(csv.field_size_limit(), original_limit)

    def test_full_image_training_rejects_tampered_stage20_method_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _full_image_config(root)
            stage20 = root / "stage20"
            output = root / "stage50"
            stage20.mkdir()
            output.mkdir()
            proposals = stage20 / "proposals.csv"
            features = stage20 / "features.csv"
            reviewed = root / "reviewed.csv"
            split = root / "split.json"
            for path in (proposals, features, reviewed, split):
                path.write_bytes(b"fixture\n")
            binding = service._stage20_proposal_config_record(
                config,
                proposals_sha256=service.sha256_file(proposals),
                raw_features_sha256=service.sha256_file(features),
            )
            binding["csv_mask_encoding"] = "tampered"
            (stage20 / "proposal_config.json").write_text(
                json.dumps(
                    binding,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="ascii",
            )
            with (
                mock.patch.object(
                    service,
                    "_require_cuda_service_device",
                    return_value="cuda",
                ),
                self.assertRaisesRegex(PublicIOError, "config/data binding changed"),
            ):
                service.train_canonical_stage(
                    config,
                    proposals,
                    reviewed,
                    split,
                    output,
                    features_path=features,
                )

    def test_service_proposal_ids_ignore_upstream_full_image_iteration_order(self) -> None:
        annotations = [
            {
                "segmentation": _rectangle_rle(
                    20,
                    20,
                    x=x,
                    y=5,
                    width=2,
                    height=2,
                ),
                "predicted_iou": 0.9,
                "stability_score": 0.9,
            }
            for x in (2, 12)
        ]
        forward = select_full_image_proposals(
            annotations,
            canvas_shape=(20, 20),
        )
        reverse = select_full_image_proposals(
            reversed(annotations),
            canvas_shape=(20, 20),
        )

        def identities(selection):
            return {
                proposal.proposal_sha256: service._proposal_id(
                    "a" * 64,
                    "sample__full.png",
                    "b" * 64,
                    0,
                    0,
                    proposal,
                    profile=FULL_IMAGE_GPU_PROFILE,
                )
                for proposal in selection.proposals
            }

        self.assertEqual(identities(forward), identities(reverse))

    def test_full_image_unit_validator_rejects_external_or_partial_units(self) -> None:
        import numpy as np

        pixels = np.zeros((20, 30, 3), dtype=np.uint8)
        prepared = _full_prepared("sample.jpg", pixels)
        self.assertIs(
            service._full_image_analysis_unit(
                prepared,
                "test",
                image_name="sample.jpg",
            ),
            prepared.tiles[0],
        )

        with self.assertRaisesRegex(PublicIOError, "exactly one"):
            service._full_image_analysis_unit(
                replace(prepared, tiles=(prepared.tiles[0], prepared.tiles[0])),
                "test",
                image_name="sample.jpg",
            )
        shifted = replace(prepared.tiles[0], x=1)
        with self.assertRaisesRegex(PublicIOError, "geometry changed"):
            service._full_image_analysis_unit(
                replace(prepared, tiles=(shifted,)),
                "test",
                image_name="sample.jpg",
            )
        tiled_evidence = dict(prepared.evidence)
        tiled_evidence["analysis_units"] = {
            **tiled_evidence["analysis_units"],
            "external_tiling": True,
        }
        with self.assertRaisesRegex(PublicIOError, "geometry changed"):
            service._full_image_analysis_unit(
                replace(prepared, evidence=tiled_evidence),
                "test",
                image_name="sample.jpg",
            )

    canvas_shape = (700, 900)
    proposal_bbox = (611, 521, 17, 11)

    def _pixels(self) -> object:
        import numpy as np

        return np.full((*self.canvas_shape, 3), (0, 220, 255), dtype=np.uint8)

    def _prepare_stage10(
        self,
        root: Path,
    ) -> tuple[SimpleNamespace, object, dict[str, object]]:
        pixels = self._pixels()
        for name in ("images", "inference_images", "assets", "stage10"):
            (root / name).mkdir()
        source = root / "images" / "card01__view.ppm"
        _write_ppm(source, width=90, height=70)
        config = _full_image_config(root)
        full_preparer = mock.Mock(
            side_effect=lambda _decoded, image_name, *, locate_grid: _full_prepared(
                image_name,
                pixels,
            )
        )
        with mock.patch.object(service, "_decode_image", return_value=pixels):
            summary = service.prepare_canonical_stage(
                config,
                root / "stage10",
                prepare_image=mock.Mock(
                    side_effect=AssertionError("full profile used tiled preparation")
                ),
                jpeg_encoder=mock.Mock(
                    side_effect=AssertionError("full profile used JPEG")
                ),
                full_image_preparer=full_preparer,
                png_encoder=lambda unit: _png_bytes(unit.pixels),
            )
        full_preparer.assert_called_once()
        self.assertTrue(full_preparer.call_args.kwargs["locate_grid"])
        return config, pixels, summary

    def _fake_raw_features(
        self,
        _image: object,
        proposals: object,
        _embedder: object,
        *,
        grid: object,
    ) -> tuple[CanonicalRawFeature, ...]:
        import numpy as np

        rows = tuple(proposals)
        self.assertEqual(len(rows), 1)
        proposal = rows[0]
        self.assertIsInstance(proposal, FullImageProposal)
        self.assertEqual(proposal.mask_crop.shape, (11, 17))
        self.assertFalse(proposal.mask_crop.flags.writeable)
        self.assertEqual((grid.tile_x, grid.tile_y), (0, 0))
        embedding = np.zeros(2048, dtype=np.float32)
        embedding[0] = 1.0
        return tuple(
            CanonicalRawFeature(
                proposal_index=proposal.proposal_index,
                scale=scale,
                predicted_iou=proposal.predicted_iou,
                stability_score=proposal.stability_score,
                values={name: 1.0 for name in CANONICAL_RAW_FEATURE_ORDER},
                embedding=embedding,
            )
            for scale in CANONICAL_FEATURE_CROP_SCALES
        )

    def _runtime(self) -> tuple[service.CanonicalStageRuntime, _FullImageRLEGenerator]:
        generator = _FullImageRLEGenerator(self.canvas_shape, self.proposal_bbox)
        runtime = service.CanonicalStageRuntime(
            amg=FullImageBackend(generator),
            embedder=_FakeEmbedder(),
            profile=FULL_IMAGE_GPU_PROFILE,
        )
        return runtime, generator

    def test_stage10_writes_one_lossless_full_frame_and_explicit_schema(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _config_value, pixels, summary = self._prepare_stage10(root)

            units_root = root / "stage10" / "processing_units"
            units = tuple(units_root.iterdir())
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].name, "card01__view__full.png")
            self.assertFalse((root / "stage10" / "tiles").exists())
            self.assertEqual(stat.S_IMODE(units[0].stat().st_mode), 0o644)
            self.assertTrue(units[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(units[0]) as decoded:
                decoded_bgr = np.asarray(decoded.convert("RGB"))[:, :, ::-1]
            self.assertTrue(np.array_equal(decoded_bgr, pixels))

            with (root / "stage10" / "tiles_index.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    service.FULL_IMAGE_UNIT_COLUMNS,
                )
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["processing_unit_kind"], "full_warped_frame")
            self.assertEqual((rows[0]["x"], rows[0]["y"]), ("0", "0"))
            self.assertEqual(
                (rows[0]["height"], rows[0]["width"]),
                tuple(str(value) for value in self.canvas_shape),
            )
            self.assertEqual(
                set(summary),
                {
                    "schema",
                    "status",
                    "profile",
                    "spatial_mode",
                    "image_count",
                    "group_count",
                    "processing_unit_count",
                    "processing_unit_kind",
                    "external_tiling",
                    "unit_format",
                    "unit_lossless",
                    "tiles_index_sha256",
                    "images",
                },
            )
            self.assertEqual(summary["schema"], "compag-curation-full-image-prepare/v1")
            self.assertEqual(summary["processing_unit_count"], 1)
            self.assertFalse(summary["external_tiling"])
            self.assertEqual(
                (summary["unit_format"], summary["unit_lossless"]), ("png", True)
            )
            self.assertEqual(summary["images"][0]["processing_unit_count"], 1)

    def test_stage20_compact_backend_features_overlay_and_profile_validator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, pixels, _summary = self._prepare_stage10(root)
            (root / "stage20").mkdir()
            runtime, generator = self._runtime()
            fake_cv2 = _FakeCV2()
            module_overrides = {
                "cv2": fake_cv2,
                "compag_curation.proposals.sam2_pipeline.gate_core": _fake_gate_core(),
            }
            with (
                mock.patch.dict(sys.modules, module_overrides),
                mock.patch.object(service, "_decode_image", return_value=pixels),
                mock.patch.object(
                    service,
                    "extract_full_image_raw_features",
                    side_effect=self._fake_raw_features,
                ),
            ):
                summary = service._generate_canonical_stage_with_dependencies(
                    config,
                    root / "stage10",
                    root / "stage20",
                    runtime_factory=lambda assets: (
                        runtime
                        if assets.profile == FULL_IMAGE_GPU_PROFILE
                        else (_ for _ in ()).throw(
                            AssertionError("full-image runtime received tiled assets")
                        )
                    ),
                    write_overlays=True,
                )

            self.assertEqual(generator.calls, 1)
            self.assertEqual(summary["profile"], FULL_IMAGE_GPU_PROFILE)
            self.assertEqual(summary["tile_count"], 1)
            self.assertTrue(summary["tile_count_compatibility_alias"])
            self.assertFalse(summary["external_tiling"])
            self.assertEqual(summary["proposal_count"], 1)
            self.assertEqual(summary["raw_feature_row_count"], 4)
            self.assertEqual(summary["mask_encoding"], service.FULL_IMAGE_MASK_ENCODING)
            self.assertEqual(summary["retained_mask_bytes"], 34)
            self.assertLess(summary["retained_mask_bytes"], 700 * 900)

            proposal_config = json.loads(
                (root / "stage20" / "proposal_config.json").read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(
                proposal_config["durable_mask_encoding"],
                FULL_IMAGE_DURABLE_MASK_ENCODING,
            )
            self.assertEqual(
                proposal_config["csv_mask_encoding"],
                service.FULL_IMAGE_MASK_ENCODING,
            )
            self.assertEqual(
                proposal_config["connected_component_policy"],
                FULL_IMAGE_COMPONENT_POLICY,
            )
            self.assertEqual(
                proposal_config["proposals_sha256"],
                summary["proposals_sha256"],
            )
            self.assertEqual(
                proposal_config["raw_features_sha256"],
                summary["raw_features_sha256"],
            )
            from compag_curation.model_bundle import (
                BUNDLE_V2_ASSET_PROFILES,
                _canonical_config_contracts,
            )

            bundle_proposal = _canonical_config_contracts(
                profile=FULL_IMAGE_GPU_PROFILE,
                asset_profile=BUNDLE_V2_ASSET_PROFILES[FULL_IMAGE_GPU_PROFILE],
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            )["proposal"]
            self.assertEqual(
                bundle_proposal["durable_mask_encoding"],
                proposal_config["durable_mask_encoding"],
            )
            self.assertEqual(
                bundle_proposal["connected_component_policy"],
                proposal_config["connected_component_policy"],
            )

            proposal_path = root / "stage20" / "proposals.csv"
            accepted = service._canonical_proposal_rows(
                proposal_path,
                profile=FULL_IMAGE_GPU_PROFILE,
            )
            self.assertEqual(len(accepted), 1)
            row = accepted[0]
            self.assertEqual(row["mask_encoding"], service.FULL_IMAGE_MASK_ENCODING)
            self.assertEqual(
                tuple(
                    int(row[name])
                    for name in (
                        "tile_bbox_x",
                        "tile_bbox_y",
                        "tile_bbox_w",
                        "tile_bbox_h",
                    )
                ),
                self.proposal_bbox,
            )
            decoded = service._decode_full_image_packed_mask(row)
            self.assertEqual(decoded.retained_bytes, 34)
            self.assertEqual(decoded.crop_array().shape, (11, 17))
            with self.assertRaisesRegex(PublicIOError, "mask encoding"):
                service._canonical_proposal_rows(
                    proposal_path,
                    profile=CANONICAL_GPU_PROFILE,
                )

            wrong_encoding_path = root / "stage20" / "wrong-encoding.csv"
            wrong_row = dict(row)
            wrong_row["mask_encoding"] = service.MASK_ENCODING
            service._write_csv_exact(
                wrong_encoding_path,
                service.CANONICAL_PROPOSAL_COLUMNS,
                [wrong_row],
            )
            with self.assertRaisesRegex(PublicIOError, "mask encoding"):
                service._canonical_proposal_rows(
                    wrong_encoding_path,
                    profile=FULL_IMAGE_GPU_PROFILE,
                )

            with (root / "stage20" / "features.csv").open(newline="") as handle:
                feature_reader = csv.DictReader(handle)
                self.assertEqual(
                    tuple(feature_reader.fieldnames or ()),
                    service.CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
                )
                self.assertEqual(len(list(feature_reader)), 4)
            overlay_manifest = json.loads(
                (root / "stage20" / "overlay_manifest.json").read_text(encoding="ascii")
            )
            self.assertEqual(
                (overlay_manifest["tile_count"], overlay_manifest["proposal_count"]),
                (1, 1),
            )
            overlay_path = root / "stage20" / overlay_manifest["tiles"][0]["overlay"]
            self.assertTrue(overlay_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(fake_cv2.contour_shapes, [(11, 17)])
            self.assertEqual(fake_cv2.contour_offsets, [(611, 521)])

    def test_full_image_inference_keeps_existing_schema_and_one_unit_alias(
        self,
    ) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            bundle_root = root / "bundle"
            output = root / "stage60"
            for path in (images, bundle_root, output):
                path.mkdir()
            _write_ppm(images / "infer01__view.ppm", width=90, height=70)
            pixels = self._pixels()
            prototype = np.zeros(2048, dtype=np.float32)
            prototype[0] = 1.0
            bundle = SimpleNamespace(
                root=bundle_root,
                schema=BUNDLE_SCHEMA_V2,
                profile=FULL_IMAGE_GPU_PROFILE,
                feature_order=CANONICAL_FEATURE_ORDER,
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
                threshold=0.5,
                classifier=b"ubj",
                imputer=(0.0,) * 93,
                prototype=tuple(float(value) for value in prototype),
                pca_mean=(0.0,) * 2048,
                pca_components=tuple((0.0,) * 2048 for _ in range(32)),
                provenance={
                    "details": {
                        "training_row_count": 32,
                        "positive_training_row_count": 1,
                    }
                },
                sam2_config=bundle_root / "sam.yaml",
                checkpoint=bundle_root / "sam.pt",
                resnet50_weights=bundle_root / "resnet.pth",
                bundle_sha256="f" * 64,
                tree_identity=(),
            )
            runtime, generator = self._runtime()
            full_prepare = mock.Mock(
                side_effect=lambda _decoded, name, *, locate_grid: _full_prepared(
                    name,
                    pixels,
                )
            )
            fake_cv2 = _FakeCV2()
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "cv2": fake_cv2,
                        "compag_curation.proposals.sam2_pipeline.gate_core": _fake_gate_core(),
                    },
                ),
                mock.patch.object(service, "_decode_image", return_value=pixels),
                mock.patch.object(service, "full_image_prepare", full_prepare),
                mock.patch.object(
                    service,
                    "canonical_prepare_images",
                    side_effect=AssertionError("full inference used external tiling"),
                ),
                mock.patch.object(
                    service,
                    "encode_full_image_png",
                    side_effect=lambda unit: _png_bytes(unit.pixels),
                ) as png_encoder,
                mock.patch.object(
                    service,
                    "encode_canonical_jpeg",
                    side_effect=AssertionError("full inference used JPEG tiles"),
                ),
                mock.patch.object(
                    service,
                    "extract_full_image_raw_features",
                    side_effect=self._fake_raw_features,
                ),
                mock.patch.object(
                    service,
                    "finalize_canonical_feature",
                    side_effect=lambda _raw, _state: {
                        name: 1.0 for name in CANONICAL_FEATURE_ORDER
                    },
                ),
            ):
                result = service._infer_canonical_bundle_with_dependencies(
                    images,
                    bundle_root,
                    output,
                    device="cuda",
                    runtime_factory=lambda assets: (
                        runtime
                        if assets.profile == FULL_IMAGE_GPU_PROFILE
                        else (_ for _ in ()).throw(
                            AssertionError("full inference received tiled assets")
                        )
                    ),
                    bundle_verifier=lambda _path: bundle,
                    predictor_loader=lambda _classifier, _imputer: object(),
                    probability_predictor=lambda _predictor, rows: np.asarray(
                        [0.9] * len(rows),
                        dtype=np.float32,
                    ),
                    nms_function=lambda _boxes, _scores, _threshold: np.asarray(
                        [0],
                        dtype=np.int64,
                    ),
                )

            full_prepare.assert_called_once()
            self.assertTrue(full_prepare.call_args.kwargs["locate_grid"])
            png_encoder.assert_called_once()
            self.assertEqual(generator.calls, 1)
            self.assertEqual(
                set(result),
                {
                    "schema",
                    "status",
                    "profile",
                    "device",
                    "bundle_sha256",
                    "feature_order_sha256",
                    "threshold",
                    "threshold_method",
                    "full_image_nms_iou",
                    "sam2_execution_points_per_batch",
                    "image_count",
                    "tile_count",
                    "proposal_count",
                    "prediction_rows",
                    "positive_rows",
                    "kept_rows",
                    "predictions_sha256",
                    "raw_feature_archive",
                    "input_inventory",
                    "paper_result_reproduction",
                },
            )
            self.assertEqual(result["schema"], "compag-curation-canonical-inference/v1")
            self.assertEqual(result["profile"], FULL_IMAGE_GPU_PROFILE)
            self.assertEqual(
                result["sam2_execution_points_per_batch"],
                FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH,
            )
            self.assertEqual(
                (
                    result["image_count"],
                    result["tile_count"],
                    result["proposal_count"],
                    result["prediction_rows"],
                    result["positive_rows"],
                    result["kept_rows"],
                ),
                (1, 1, 1, 1, 1, 1),
            )
            self.assertEqual(result["input_inventory"][0]["tile_count"], 1)
            self.assertEqual(
                result["raw_feature_archive"]["profile"],
                FULL_IMAGE_GPU_PROFILE,
            )
            with (output / "predictions.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    service.CANONICAL_INFERENCE_COLUMNS,
                )
                prediction_rows = list(reader)
            self.assertEqual(len(prediction_rows), 1)
            self.assertEqual(prediction_rows[0]["full_image"], "infer01__view.ppm")
            self.assertEqual(prediction_rows[0]["tile_name"], "infer01__view__full.png")


if __name__ == "__main__":
    unittest.main()
