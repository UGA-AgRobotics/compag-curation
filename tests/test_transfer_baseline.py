from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
)
from compag_curation.canonical.transfer_baseline import (
    EXPECTED_R92_TRANSFER_SOURCE_SHA256,
    EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES,
    LEGACY_TRANSFER_COLUMNS,
    TRANSFER_BASELINE_COLUMNS,
    import_legacy_transfer_baseline,
    import_r92_transfer_baseline,
    iter_transfer_baseline_rows,
    verify_transfer_baseline,
)
from compag_curation.public_io import PublicIOError, canonical_json_value
from compag_curation.review.published_model import (
    PRESET_ID,
    PRESET_RESOURCE_MANIFEST_SHA256,
)


class TransferBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _legacy_row(
        *,
        file_name: str,
        annotation_id: int,
        scale: float,
        label: int,
        reviewed: bool,
        action: str,
        serial: int,
    ) -> dict[str, str]:
        row = {name: "0" for name in LEGACY_TRANSFER_COLUMNS}
        row.update(
            {
                "file_name": file_name,
                "ann_id": str(annotation_id),
                "scale": format(scale, ".4f"),
                "reviewed": "1" if reviewed else "0",
                "review_tag": action,
                "class_id": "1" if label else "2",
                "class_name": "CJ" if label else "non-CJ",
                "label": str(label),
            }
        )
        for index, name in enumerate(CANONICAL_FEATURE_ORDER, start=1):
            row[name] = str(serial * 1000 + index)
        return row

    def _source(
        self,
        name: str = "legacy.csv",
        *,
        duplicate_scale: bool = False,
        inconsistent_action: bool = False,
    ) -> Path:
        path = self.root / name
        rows: list[dict[str, str]] = []
        annotation_id = 1
        actions = ("accept", "flip", "sus_accept", "sus_flip")
        for group_index in range(8):
            file_name = f"IMG_{9400 + group_index}_y00000x00000.jpg"
            for label in (0, 1):
                reviewed = label == 1
                action = actions[group_index % len(actions)] if reviewed else ""
                scales = list(CANONICAL_FEATURE_CROP_SCALES)
                if duplicate_scale and group_index == 0 and label == 0:
                    scales[1] = scales[0]
                for scale_index, scale in enumerate(scales):
                    row_action = action
                    if (
                        inconsistent_action
                        and group_index == 0
                        and label == 1
                        and scale_index == 3
                    ):
                        row_action = "flip" if action != "flip" else "accept"
                    rows.append(
                        self._legacy_row(
                            file_name=file_name,
                            annotation_id=annotation_id,
                            scale=scale,
                            label=label,
                            reviewed=reviewed,
                            action=row_action,
                            serial=annotation_id,
                        )
                    )
                annotation_id += 1

        # Both review states can be incomplete.  They are audited and omitted.
        for reviewed, action, group_index in (
            (False, "", 0),
            (True, "accept", 1),
        ):
            for scale in CANONICAL_FEATURE_CROP_SCALES[1:]:
                rows.append(
                    self._legacy_row(
                        file_name=f"IMG_{9400 + group_index}_y00512x00000.jpg",
                        annotation_id=annotation_id,
                        scale=scale,
                        label=0,
                        reviewed=reviewed,
                        action=action,
                        serial=annotation_id,
                    )
                )
            annotation_id += 1

        # A skip-only source group remains in the all-source leakage ledger.
        for scale in CANONICAL_FEATURE_CROP_SCALES:
            rows.append(
                self._legacy_row(
                    file_name="IMG_9475_y00000x00000.jpg",
                    annotation_id=annotation_id,
                    scale=scale,
                    label=0,
                    reviewed=True,
                    action="skip",
                    serial=annotation_id,
                )
            )

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=LEGACY_TRANSFER_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        value = canonical_json_value(path.read_bytes(), path.name)
        assert isinstance(value, dict)
        return value

    def test_import_preserves_fit_population_metadata_and_seals_split(self) -> None:
        source = self._source()
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        baseline = import_legacy_transfer_baseline(source, self.root / "baseline")

        self.assertEqual(baseline.source_sha256, source_sha256)
        self.assertEqual((baseline.proposal_count, baseline.row_count), (16, 64))
        self.assertEqual(len(baseline.source_groups), 9)
        self.assertIn("img_9475", baseline.source_groups)
        self.assertNotIn("img_9475", baseline.train_groups | baseline.test_groups)
        self.assertFalse(baseline.train_groups & baseline.test_groups)
        self.assertEqual(
            baseline.train_groups | baseline.test_groups,
            frozenset(f"img_{9400 + index}" for index in range(8)),
        )

        manifest = dict(baseline.manifest)
        selection = manifest["selection"]
        self.assertEqual(
            selection["policy"],
            "ALL_LABELED_NON_SKIP_COMPLETE_CANONICAL_SCALE_BLOCKS",
        )
        self.assertEqual(selection["reviewed_proposal_count"], 10)
        self.assertEqual(selection["unreviewed_proposal_count"], 9)
        self.assertEqual(selection["skipped_proposal_count"], 1)
        self.assertEqual(selection["non_skip_proposal_count"], 18)
        self.assertEqual(selection["imported_reviewed_proposal_count"], 8)
        self.assertEqual(selection["imported_unreviewed_proposal_count"], 8)
        self.assertEqual(
            selection["exclusions"],
            [
                {
                    "reason": "LABELED_NON_SKIP_PROPOSAL_LACKS_EXACT_FOUR_CANONICAL_SCALES",
                    "proposal_count": 2,
                    "source_row_count": 6,
                    "reviewed_proposal_count": 1,
                    "unreviewed_proposal_count": 1,
                }
            ],
        )
        self.assertEqual(
            manifest["published_model_binding"],
            {
                "preset_id": PRESET_ID,
                "resource_manifest_sha256": PRESET_RESOURCE_MANIFEST_SHA256,
                "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                "feature_state_source": "BUNDLED_PUBLISHED_MODEL_PRESET_NOT_LEGACY_CSV",
                "legacy_csv_feature_state_sha256": None,
                "legacy_csv_prototype_sha256": None,
                "legacy_csv_pca_sha256": None,
            },
        )
        self.assertFalse(manifest["claims"]["historical_fit_time_split_established"])
        self.assertEqual(manifest["claims"]["paper_result_reproduction"], "NOT_CLAIMED")

        with baseline.features_path.open(encoding="utf-8", newline="") as handle:
            header = tuple(next(csv.reader(handle)))
        self.assertEqual(header, TRANSFER_BASELINE_COLUMNS)
        self.assertEqual(
            header[4 : 4 + len(CANONICAL_FEATURE_ORDER)],
            CANONICAL_FEATURE_ORDER,
        )

        rows = list(iter_transfer_baseline_rows(baseline))
        by_proposal: dict[str, list[object]] = defaultdict(list)
        labels_by_group: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            by_proposal[row.proposal_id].append(row)
            labels_by_group[row.group_id].add(row.label)
            if row.reviewed:
                expected = 0.4 if row.review_action.startswith("sus_") else 1.0
                self.assertEqual(row.review_weight, expected)
            else:
                self.assertEqual((row.review_action, row.review_weight), ("", 1.0))
        self.assertEqual(len(by_proposal), 16)
        self.assertTrue(
            all(
                tuple(item.scale for item in block)
                == tuple(CANONICAL_FEATURE_CROP_SCALES)
                for block in by_proposal.values()
            )
        )
        self.assertEqual(set(labels_by_group), set(baseline.train_groups | baseline.test_groups))
        self.assertTrue(all(labels == {0, 1} for labels in labels_by_group.values()))

        self.assertEqual(len(baseline.folds), 5)
        validation_union: set[str] = set()
        for fold in baseline.folds:
            self.assertFalse(fold.train_groups & fold.validation_groups)
            self.assertEqual(
                fold.train_groups | fold.validation_groups,
                baseline.train_groups,
            )
            validation_union.update(fold.validation_groups)
            for partition in (fold.train_groups, fold.validation_groups):
                labels = {
                    row.label
                    for row in rows
                    if row.group_id in partition
                }
                self.assertEqual(labels, {0, 1})
        self.assertEqual(validation_union, set(baseline.train_groups))

        serialized = b"\n".join(
            path.read_bytes() for path in sorted(baseline.root.iterdir())
        )
        self.assertNotIn(str(source).encode(), serialized)
        self.assertNotIn(b'"absolute_source_path"', serialized)
        self.assertNotIn(b"ann_id", baseline.features_path.read_bytes().splitlines()[0])
        self.assertEqual(verify_transfer_baseline(baseline.root), baseline)

    def test_import_is_deterministic_atomic_and_no_clobber(self) -> None:
        source = self._source()
        first = import_legacy_transfer_baseline(source, self.root / "first")
        second = import_legacy_transfer_baseline(source, self.root / "second")
        for name in ("features.csv", "source_groups.json", "split.json", "manifest.json"):
            self.assertEqual(
                (first.root / name).read_bytes(),
                (second.root / name).read_bytes(),
            )
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first.root.iterdir()
        }
        with self.assertRaisesRegex(PublicIOError, "already exists"):
            import_legacy_transfer_baseline(source, first.root)
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first.root.iterdir()
        }
        self.assertEqual(after, before)

        strict_output = self.root / "strict-r92"
        with self.assertRaisesRegex(PublicIOError, "required SHA-256 and size"):
            import_r92_transfer_baseline(source, strict_output)
        self.assertFalse(strict_output.exists())
        self.assertEqual(len(EXPECTED_R92_TRANSFER_SOURCE_SHA256), 64)
        self.assertEqual(EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES, 582_968_556)

    def test_conflicts_and_duplicate_scales_fail_without_publication(self) -> None:
        for option, message in (
            ({"duplicate_scale": True}, "duplicate crop scale"),
            ({"inconsistent_action": True}, "inconsistent review"),
        ):
            with self.subTest(option=option):
                source = self._source(f"bad-{len(list(self.root.iterdir()))}.csv", **option)
                output = self.root / f"failed-{len(list(self.root.iterdir()))}"
                with self.assertRaisesRegex(PublicIOError, message):
                    import_legacy_transfer_baseline(source, output)
                self.assertFalse(output.exists())
                self.assertEqual(
                    list(self.root.glob(f".{output.name}.transfer-baseline-*")),
                    [],
                )

    def test_exact_header_and_tamper_are_rejected(self) -> None:
        source = self._source()
        wrong = self.root / "wrong.csv"
        payload = source.read_text(encoding="utf-8").replace(
            "file_name,ann_id,scale",
            "ann_id,file_name,scale",
            1,
        )
        wrong.write_text(payload, encoding="utf-8")
        with self.assertRaisesRegex(PublicIOError, "exact 109-column"):
            import_legacy_transfer_baseline(wrong, self.root / "wrong-output")

        baseline = import_legacy_transfer_baseline(source, self.root / "baseline")
        with baseline.features_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write("tamper\n")
        with self.assertRaises(PublicIOError):
            verify_transfer_baseline(baseline.root)


if __name__ == "__main__":
    unittest.main()
