from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _rle_from_points(
    height: int,
    width: int,
    points: set[tuple[int, int]],
) -> dict[str, object]:
    positions = sorted(x * height + y for y, x in points)
    if not positions:
        return {"size": [height, width], "counts": [height * width]}
    runs: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        runs.append((start, previous + 1))
        start = previous = position
    runs.append((start, previous + 1))
    counts: list[int] = []
    cursor = 0
    for start, stop in runs:
        counts.extend((start - cursor, stop - start))
        cursor = stop
    counts.append(height * width - cursor)
    return {"size": [height, width], "counts": counts}


def _rectangle_rle(
    height: int,
    width: int,
    *,
    x: int,
    y: int,
    box_width: int,
    box_height: int,
) -> dict[str, object]:
    return _rle_from_points(
        height,
        width,
        {
            (row, column)
            for row in range(y, y + box_height)
            for column in range(x, x + box_width)
        },
    )


def _annotation(
    rle: dict[str, object],
    *,
    predicted_iou: float = 0.9,
    stability_score: float = 0.9,
) -> dict[str, object]:
    return {
        "segmentation": rle,
        "predicted_iou": predicted_iou,
        "stability_score": stability_score,
    }


class FullImageAMGConstructionTests(unittest.TestCase):
    def test_build_uses_native_crop_rle_and_memory_safe_batch(self) -> None:
        from compag_curation.canonical.full_image import build_full_image_amg

        model = object()
        captured: dict[str, object] = {}
        sentinel = SimpleNamespace(generate=lambda _image: ())

        def factory(received_model, **kwargs):
            captured["model"] = received_model
            captured["kwargs"] = kwargs
            return sentinel

        with (
            patch(
                "compag_curation.canonical.full_image._attest_full_image_sam2_cuda_model",
                return_value=0,
            ) as attest_model,
            patch(
                "compag_curation.canonical.full_image._attest_full_image_amg_cuda",
                return_value=0,
            ) as attest_generator,
        ):
            generator = build_full_image_amg(model, generator_factory=factory)

        self.assertIs(generator, sentinel)
        self.assertIs(captured["model"], model)
        kwargs = captured["kwargs"]
        self.assertEqual(kwargs["points_per_side"], 64)
        self.assertEqual(kwargs["points_per_batch"], 8)
        self.assertEqual(kwargs["pred_iou_thresh"], 0.80)
        self.assertEqual(kwargs["stability_score_thresh"], 0.88)
        self.assertEqual(kwargs["crop_n_layers"], 2)
        self.assertEqual(kwargs["crop_n_points_downscale_factor"], 1)
        self.assertEqual(kwargs["crop_overlap_ratio"], 512 / 1500)
        self.assertEqual(kwargs["box_nms_thresh"], 0.70)
        self.assertEqual(kwargs["crop_nms_thresh"], 0.70)
        self.assertEqual(kwargs["output_mode"], "uncompressed_rle")
        self.assertFalse(kwargs["use_m2m"])
        self.assertTrue(kwargs["multimask_output"])
        attest_model.assert_called_once_with(model)
        attest_generator.assert_called_once_with(sentinel, expected_model=model)

    def test_execution_batch_can_only_be_reduced_from_safe_baseline(self) -> None:
        from compag_curation.canonical.full_image import build_full_image_amg

        captured: dict[str, object] = {}

        def factory(_model, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(generate=lambda _image: ())

        with (
            patch(
                "compag_curation.canonical.full_image._attest_full_image_sam2_cuda_model",
                return_value=0,
            ),
            patch(
                "compag_curation.canonical.full_image._attest_full_image_amg_cuda",
                return_value=0,
            ),
        ):
            build_full_image_amg(
                object(), execution_points_per_batch=3, generator_factory=factory
            )
        self.assertEqual(captured["points_per_batch"], 3)
        with self.assertRaisesRegex(ValueError, "memory-safe"):
            build_full_image_amg(
                object(), execution_points_per_batch=9, generator_factory=factory
            )

    def test_memory_bounded_subclass_drops_only_dead_non_m2m_logits(self) -> None:
        from compag_curation.canonical.full_image import (
            _memory_bounded_sam2_amg_type,
        )

        class Upstream:
            def __init__(self, use_m2m: bool) -> None:
                self.use_m2m = use_m2m

            def _process_batch(self):
                return {"rles": [1], "low_res_masks": object()}

        bounded_type = _memory_bounded_sam2_amg_type(Upstream)
        without_m2m = bounded_type(False)._process_batch()
        with_m2m = bounded_type(True)._process_batch()

        self.assertEqual(without_m2m, {"rles": [1]})
        self.assertIn("low_res_masks", with_m2m)


class FullImagePackedMaskTests(unittest.TestCase):
    def test_large_canvas_retains_only_bbox_packbits_and_independent_hash(self) -> None:
        from compag_curation.canonical.full_image import (
            full_image_proposal_sha256,
            pack_uncompressed_rle,
        )

        rle = _rectangle_rle(
            4000,
            6000,
            x=4321,
            y=1234,
            box_width=9,
            box_height=10,
        )
        packed = pack_uncompressed_rle(rle)

        self.assertIsNotNone(packed)
        assert packed is not None
        self.assertEqual(packed.bbox, (4321, 1234, 9, 10))
        self.assertEqual(packed.area, 90)
        self.assertEqual(packed.column_stride, 2)
        self.assertEqual(packed.retained_bytes, 18)
        self.assertLess(packed.retained_bytes, (4000 * 6000) // 1_000_000)
        proposal_hash = full_image_proposal_sha256(packed)
        self.assertEqual(len(proposal_hash), 64)
        self.assertNotEqual(proposal_hash, packed.mask_sha256)
        self.assertEqual(proposal_hash, full_image_proposal_sha256(packed))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required to decode a mask crop")
    def test_decoded_crop_is_uint8_bbox_sized_and_read_only(self) -> None:
        import numpy as np

        from compag_curation.canonical.full_image import pack_uncompressed_rle

        points = {(2, 3), (2, 4), (3, 4), (4, 5)}
        packed = pack_uncompressed_rle(_rle_from_points(10, 12, points))

        assert packed is not None
        crop = packed.crop_array()
        expected = np.asarray(
            [
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )
        self.assertEqual(packed.bbox, (3, 2, 3, 3))
        self.assertEqual(crop.dtype, np.uint8)
        self.assertEqual(crop.shape, (3, 3))
        self.assertTrue(np.array_equal(crop, expected))
        self.assertFalse(crop.flags.writeable)
        with self.assertRaises(ValueError):
            crop[0, 0] = 0

    def test_compact_intersection_is_exact_across_bbox_offsets(self) -> None:
        from compag_curation.canonical.full_image import pack_uncompressed_rle

        left = pack_uncompressed_rle(
            _rectangle_rle(20, 20, x=2, y=3, box_width=4, box_height=5)
        )
        right = pack_uncompressed_rle(
            _rectangle_rle(20, 20, x=4, y=5, box_width=5, box_height=4)
        )

        assert left is not None and right is not None
        self.assertEqual(left.intersection_area(right), 6)
        self.assertAlmostEqual(left.iou(right), 6 / 34)

    def test_claimed_area_must_equal_packed_payload_popcount(self) -> None:
        from compag_curation.canonical.full_image import (
            PackedMask,
            _packed_mask_sha256,
            pack_uncompressed_rle,
        )

        payload = bytes((1, 0))
        claimed_area = 2
        digest = _packed_mask_sha256(
            canvas_height=8,
            canvas_width=8,
            origin_x=2,
            origin_y=3,
            width=2,
            height=1,
            area=claimed_area,
            column_stride=1,
            packed=payload,
        )
        with self.assertRaisesRegex(ValueError, "area does not match"):
            PackedMask(
                canvas_height=8,
                canvas_width=8,
                origin_x=2,
                origin_y=3,
                width=2,
                height=1,
                area=claimed_area,
                column_stride=1,
                packed=payload,
                mask_sha256=digest,
            )

    def test_disconnected_rle_is_canonicalized_to_largest_component(self) -> None:
        from compag_curation.canonical.full_image import pack_uncompressed_rle

        packed = pack_uncompressed_rle(
            _rle_from_points(
                12,
                12,
                {(1, 1), (1, 2), (2, 1), (2, 2), (9, 9)},
            )
        )
        assert packed is not None
        self.assertEqual(packed.bbox, (1, 1, 2, 2))
        self.assertEqual(packed.area, 4)
        self.assertEqual(sum(byte.bit_count() for byte in packed.packed), 4)

    def test_nonminimal_zero_bordered_crop_is_rejected(self) -> None:
        from compag_curation.canonical.full_image import (
            PackedMask,
            _packed_mask_sha256,
        )

        payload = bytes((0, 2, 0))
        digest = _packed_mask_sha256(
            canvas_height=8,
            canvas_width=8,
            origin_x=1,
            origin_y=1,
            width=3,
            height=3,
            area=1,
            column_stride=1,
            packed=payload,
        )
        with self.assertRaisesRegex(ValueError, "bbox is not minimal"):
            PackedMask(
                canvas_height=8,
                canvas_width=8,
                origin_x=1,
                origin_y=1,
                width=3,
                height=3,
                area=1,
                column_stride=1,
                packed=payload,
                mask_sha256=digest,
            )

    def test_rejects_compressed_or_wrong_canvas_rle(self) -> None:
        from compag_curation.canonical.full_image import pack_uncompressed_rle

        with self.assertRaisesRegex(ValueError, "uncompressed sequence"):
            pack_uncompressed_rle({"size": [4, 4], "counts": "encoded"})
        with self.assertRaisesRegex(ValueError, "height differs"):
            pack_uncompressed_rle(
                {"size": [4, 4], "counts": [16]},
                canvas_height=5,
                canvas_width=4,
            )
        with self.assertRaisesRegex(ValueError, "do not cover"):
            pack_uncompressed_rle({"size": [4, 4], "counts": [15]})


class FullImageSelectionTests(unittest.TestCase):
    def test_higher_score_duplicate_wins_and_attributes_are_exposed(self) -> None:
        from compag_curation.canonical.full_image import select_full_image_proposals

        same = _rectangle_rle(30, 40, x=5, y=7, box_width=6, box_height=8)
        other = _rectangle_rle(30, 40, x=25, y=20, box_width=3, box_height=4)
        result = select_full_image_proposals(
            [
                _annotation(same, predicted_iou=0.80, stability_score=0.88),
                _annotation(same, predicted_iou=0.95, stability_score=0.91),
                _annotation(other, predicted_iou=0.90, stability_score=0.90),
            ],
            canvas_shape=(30, 40),
        )

        self.assertEqual(result.input_count, 3)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(len(result.proposals), 2)
        first = result.proposals[0]
        self.assertEqual(first.source_index, 0)
        self.assertEqual(first.proposal_index, 1)
        self.assertEqual(first.bbox, (5, 7, 6, 8))
        self.assertEqual(first.area, 48)
        self.assertEqual(len(first.mask_sha256), 64)
        self.assertEqual(len(first.proposal_sha256), 64)
        self.assertNotEqual(first.mask_sha256, first.proposal_sha256)

    def test_bbox_prefilter_precedes_exact_mask_iou(self) -> None:
        from compag_curation.canonical.full_image import select_full_image_proposals

        first = _rectangle_rle(40, 40, x=2, y=2, box_width=5, box_height=5)
        shifted = _rectangle_rle(40, 40, x=3, y=2, box_width=5, box_height=5)
        distant = _rectangle_rle(40, 40, x=30, y=30, box_width=3, box_height=3)
        result = select_full_image_proposals(
            [
                _annotation(first, predicted_iou=0.95),
                _annotation(shifted, predicted_iou=0.90),
                _annotation(distant, predicted_iou=0.85),
            ],
            canvas_shape=(40, 40),
            merge_iou_threshold=0.60,
        )

        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(len(result.proposals), 2)
        self.assertEqual(result.mask_comparison_count, 1)
        self.assertEqual(result.bbox_prefilter_count, 1)

    def test_hash_tie_break_makes_cap_independent_of_input_order(self) -> None:
        from compag_curation.canonical.full_image import select_full_image_proposals

        annotations = [
            _annotation(_rectangle_rle(20, 20, x=x, y=10, box_width=1, box_height=1))
            for x in (1, 5, 9, 13)
        ]
        forward = select_full_image_proposals(
            annotations,
            canvas_shape=(20, 20),
            max_proposals=2,
        )
        reverse = select_full_image_proposals(
            reversed(annotations),
            canvas_shape=(20, 20),
            max_proposals=2,
        )

        self.assertEqual(
            [row.proposal_sha256 for row in forward.proposals],
            [row.proposal_sha256 for row in reverse.proposals],
        )
        self.assertEqual(
            [row.source_index for row in forward.proposals],
            [row.source_index for row in reverse.proposals],
        )
        self.assertEqual([row.proposal_index for row in forward.proposals], [1, 2])
        self.assertEqual(forward.capped_count, 2)
        self.assertEqual(reverse.capped_count, 2)

    def test_duplicates_after_cap_are_not_misreported_as_capped(self) -> None:
        from compag_curation.canonical.full_image import select_full_image_proposals

        first = _rectangle_rle(20, 20, x=2, y=2, box_width=2, box_height=2)
        second = _rectangle_rle(20, 20, x=12, y=12, box_width=2, box_height=2)
        result = select_full_image_proposals(
            [
                _annotation(first, predicted_iou=0.99),
                _annotation(second, predicted_iou=0.98),
                _annotation(first, predicted_iou=0.97),
            ],
            canvas_shape=(20, 20),
            max_proposals=1,
        )

        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.capped_count, 1)

        duplicate_of_capped = select_full_image_proposals(
            [
                _annotation(first, predicted_iou=0.99),
                _annotation(second, predicted_iou=0.98),
                _annotation(second, predicted_iou=0.97),
            ],
            canvas_shape=(20, 20),
            max_proposals=1,
        )
        self.assertEqual(len(duplicate_of_capped.proposals), 1)
        self.assertEqual(duplicate_of_capped.duplicate_count, 1)
        self.assertEqual(duplicate_of_capped.capped_count, 1)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required by the extractor contract")
    def test_backend_generate_returns_selection_with_read_only_crop(self) -> None:
        import numpy as np

        from compag_curation.canonical.full_image import (
            FullImageBackend,
            FullImageProposalSelection,
        )

        image = np.zeros((50, 60, 3), dtype=np.uint8)

        class FakeGenerator:
            def __init__(self) -> None:
                self.received = None
                self.predictor = object()

            def generate(self, received):
                self.received = received
                return [
                    _annotation(
                        _rectangle_rle(
                            50,
                            60,
                            x=11,
                            y=13,
                            box_width=7,
                            box_height=5,
                        )
                    )
                ]

        generator = FakeGenerator()
        backend = FullImageBackend(generator)
        selection = backend.generate(image)

        self.assertIsInstance(selection, FullImageProposalSelection)
        self.assertIs(backend.generator, generator)
        self.assertIs(backend.predictor, generator.predictor)
        self.assertIs(generator.received, image)
        self.assertEqual(len(selection.proposals), 1)
        proposal = selection.proposals[0]
        self.assertEqual(proposal.mask_crop.shape, (5, 7))
        self.assertEqual(proposal.mask_crop.dtype, np.uint8)
        self.assertFalse(proposal.mask_crop.flags.writeable)
        self.assertEqual(selection.retained_mask_bytes, 7)


if __name__ == "__main__":
    unittest.main()
