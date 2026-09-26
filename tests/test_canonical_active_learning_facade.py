from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation import model_bundle
from compag_curation.canonical import active_learning_facade as facade
from compag_curation.canonical import active_learning_service as adapter
from compag_curation.canonical import service
from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalRawFeature,
    canonical_raw_feature_csv_row,
)
from compag_curation.canonical.proposals import canonical_mask_sha256
from compag_curation.canonical.serialization import (
    portable_classifier_bytes,
    serialize_canonical_feature_state,
)
from compag_curation.canonical.spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_FULL_IMAGE_NMS_IOU,
    CANONICAL_GPU_PROFILE,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
    CANONICAL_PROFILE,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    amg_settings_for_profile,
    stage20_amg_execution_points_per_batch,
)
from compag_curation.model_bundle import (
    BUNDLE_V2_ASSET_PROFILES,
    BundleV2WriteRequest,
    _canonical_config_contracts,
    verify_model_bundle,
    write_model_bundle_v2,
)
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    compact_json_sha256,
    manifest_rows,
    owned_manifest_contract,
    sha256_file,
    write_new_json,
)
from compag_curation.run_state import STAGE_REQUIRED_FILES, STAGE_SCHEMA
from tests.test_canonical_active_learning_service import _classifier, _state


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _fixture_mask_sha256() -> str:
    import numpy as np

    return canonical_mask_sha256(np.ones((1, 1), dtype=np.uint8))


def _raw_row(
    *,
    name: str,
    group: str,
    label: int,
    proposal_index: int,
    scale: float,
) -> dict[str, object]:
    import numpy as np

    proposal_id = _sha(f"proposal:{name}")
    image_name = f"{group}__{name}.jpg"
    image_sha256 = _sha(f"image-bytes:{name}")
    image_id = hashlib.sha256(
        b"compag-image-v1\0"
        + image_name.encode("utf-8")
        + b"\0"
        + group.encode("ascii")
        + b"\0"
        + image_sha256.encode("ascii")
    ).hexdigest()
    embedding = np.zeros(2048, dtype=np.float32)
    embedding[label] = 1.0
    embedding.setflags(write=False)
    identity = {
        "proposal_id": proposal_id,
        "proposal_sha256": proposal_id,
        "image_id": image_id,
        "image_sha256": image_sha256,
        "image_name": image_name,
        "group_id": group,
        "tile_name": f"{name}_y00000x00000.jpg",
        "tile_sha256": _sha(f"tile:{name}"),
        "tile_x": "0",
        "tile_y": "0",
        "mask_sha256": _fixture_mask_sha256(),
    }
    scientific = canonical_raw_feature_csv_row(
        CanonicalRawFeature(
            proposal_index=proposal_index,
            scale=scale,
            predicted_iou=0.9,
            stability_score=0.95,
            values={
                feature: float(proposal_index) / 101.0
                for feature in CANONICAL_RAW_FEATURE_ORDER
            },
            embedding=embedding,
        )
    )
    rendered = {key: str(value) for key, value in scientific.items()}
    row = {**identity, **rendered}
    if tuple(row) != service.CANONICAL_RAW_FEATURE_TABLE_COLUMNS:
        raise AssertionError("fixture raw-feature schema changed")
    return row


def _reviewed_copy(
    request: Path,
    output: Path,
    labels: dict[str, int],
) -> None:
    with request.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            row["label"] = str(labels[row["proposal_id"]])
            row["review_action"] = "accept"
            row["review_weight"] = "1.0"
            row["review_status"] = "reviewed"
            writer.writerow(row)


class CanonicalActiveLearningFacadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        payloads = {
            "hiera.yaml": b"fixture-hiera-l-config\n",
            "hiera.pt": b"fixture-hiera-l-checkpoint\n",
            "sam-license.txt": b"fixture-sam-license\n",
            "resnet50.pth": b"fixture-resnet50-weights\n",
            "resnet-license.txt": b"fixture-resnet-license\n",
        }
        for name, payload in payloads.items():
            (self.assets / name).write_bytes(payload)
        digests = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
        self.asset_profile = {
            "sam2_architecture": "sam2.1_hiera_large",
            "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "sam2_checkpoint_sha256": digests["hiera.pt"],
            "sam2_config_sha256": digests["hiera.yaml"],
            "sam2_license_sha256": digests["sam-license.txt"],
            "resnet50_architecture": "torchvision.models.resnet50",
            "resnet50_weights_identity": "ResNet50_Weights.IMAGENET1K_V2",
            "resnet50_weights_sha256": digests["resnet50.pth"],
            "resnet50_license_sha256": digests["resnet-license.txt"],
        }
        self.efficient_asset_profile = dict(
            BUNDLE_V2_ASSET_PROFILES[EFFICIENT_GPU_PROFILE]
        )
        patcher = mock.patch.dict(
            BUNDLE_V2_ASSET_PROFILES,
            {
                CANONICAL_PROFILE: self.asset_profile,
                CANONICAL_GPU_PROFILE: self.asset_profile,
                EFFICIENT_GPU_PROFILE: self.efficient_asset_profile,
                FULL_IMAGE_GPU_PROFILE: self.asset_profile,
            },
            clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        gpu_lock_patcher = mock.patch.object(
            model_bundle,
            "science_gpu_lock_identity",
            return_value={"sha256": model_bundle.SCIENCE_GPU_LOCK_SHA256},
        )
        gpu_lock_patcher.start()
        self.addCleanup(gpu_lock_patcher.stop)
        self.science_dependencies = {
            "schema": "compag-curation-test-science-dependencies/v1",
            "sha256": _sha("locked-science-environment"),
        }
        science_patcher = mock.patch.object(
            facade,
            "require_science_dependencies",
            return_value=self.science_dependencies,
        )
        self.science_preflight = science_patcher.start()
        self.addCleanup(science_patcher.stop)
        self.stage20, self.initial_labels = self._stage20()

    def _seal_stage20(self, root: Path) -> None:
        success = root / "_SUCCESS.json"
        if success.exists():
            success.unlink()
        files, directories = owned_manifest_contract(
            root,
            excluded={"_SUCCESS.json"},
        )
        write_new_json(
            success,
            {
                "schema": STAGE_SCHEMA,
                "status": "PASS",
                "run_id": str(uuid.uuid4()),
                "stage": "20_proposals_features",
                "completed_at_utc": "2026-08-30T00:00:00Z",
                "required": list(STAGE_REQUIRED_FILES["20_proposals_features"]),
                "files": files,
                "files_sha256": compact_json_sha256(files),
                "directories": directories,
                "directories_sha256": compact_json_sha256(directories),
            },
        )

    def _stage20(
        self,
        *,
        profile: str = CANONICAL_GPU_PROFILE,
        stage_name: str = "stage20",
    ) -> tuple[Path, dict[str, int]]:
        root = self.root / stage_name
        root.mkdir(mode=0o700)
        rows: list[dict[str, object]] = []
        proposal_rows: list[dict[str, object]] = []
        review_rows: list[dict[str, object]] = []
        labels: dict[str, int] = {}
        proposal_index = 0
        for group_index in range(6):
            for label in (0, 1):
                proposal_index += 1
                name = f"initial-{group_index}-{label}"
                proposal_id = _sha(f"proposal:{name}")
                labels[proposal_id] = label
                identity = _raw_row(
                    name=name,
                    group=f"g{group_index}",
                    label=label,
                    proposal_index=proposal_index,
                    scale=CANONICAL_FEATURE_CROP_SCALES[0],
                )
                proposal_rows.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_sha256": proposal_id,
                        "image_id": identity["image_id"],
                        "image_sha256": identity["image_sha256"],
                        "image_name": identity["image_name"],
                        "group_id": identity["group_id"],
                        "tile_name": identity["tile_name"],
                        "tile_sha256": identity["tile_sha256"],
                        "tile_x": "0",
                        "tile_y": "0",
                        "source_index": str(proposal_index),
                        "proposal_index": str(proposal_index),
                        "mask_sha256": identity["mask_sha256"],
                        "mask_area": "1",
                        "mask_encoding": service.MASK_ENCODING,
                        "mask_height": "1",
                        "mask_width": "1",
                        "mask_packbits_base64": "AQ==",
                        "tile_bbox_x": "0",
                        "tile_bbox_y": "0",
                        "tile_bbox_w": "1",
                        "tile_bbox_h": "1",
                        "bbox_x": "0",
                        "bbox_y": "0",
                        "bbox_w": "1",
                        "bbox_h": "1",
                        "poly": "[0,0,1,0,1,1,0,1]",
                        "predicted_iou": service._format_float(0.9),
                        "stability_score": service._format_float(0.95),
                    }
                )
                review_rows.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_sha256": proposal_id,
                        "image_id": identity["image_id"],
                        "image_sha256": identity["image_sha256"],
                        "group_id": identity["group_id"],
                        "label": "",
                        "review_action": "",
                        "review_weight": "",
                        "review_status": "pending",
                    }
                )
                for scale in CANONICAL_FEATURE_CROP_SCALES:
                    rows.append(
                        _raw_row(
                            name=name,
                            group=f"g{group_index}",
                            label=label,
                            proposal_index=proposal_index,
                            scale=scale,
                        )
                    )
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                str(row["proposal_id"]),
                tuple(CANONICAL_FEATURE_CROP_SCALES).index(float(row["scale"])),
            ),
        )
        service._write_csv_exact(
            root / "proposals.csv",
            service.CANONICAL_PROPOSAL_COLUMNS,
            sorted(proposal_rows, key=lambda row: str(row["proposal_id"])),
        )
        service._write_csv_exact(
            root / "features.csv",
            service.CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
            ordered_rows,
        )
        service._write_csv_exact(
            root / "review_request.csv",
            service.CANONICAL_REVIEW_COLUMNS,
            sorted(review_rows, key=lambda row: str(row["proposal_id"])),
        )
        settings = amg_settings_for_profile(profile)
        assets = BUNDLE_V2_ASSET_PROFILES[profile]
        microbatch = stage20_amg_execution_points_per_batch(profile)
        proposal_config = {
            "schema": "compag-curation-canonical-proposal-config/v1",
            "status": "PASS",
            **settings.provenance_record(profile=profile),
            "sam2_execution_points_per_batch": microbatch,
            "sam2_checkpoint_sha256": assets["sam2_checkpoint_sha256"],
            "sam2_config_sha256": assets["sam2_config_sha256"],
            "sam2_config_locator": assets["sam2_config_locator"],
            "resnet50_weights_sha256": assets["resnet50_weights_sha256"],
            "embedding_backbone": "resnet50-imagenet1k-v2",
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "masked_crop_padding": 0.10,
            "pca_fit": "DEFERRED_UNTIL_REVIEWED_TRAIN_PARTITION",
            "prototype_fit": "DEFERRED_UNTIL_REVIEWED_TRAIN_POSITIVES",
            "raw_feature_columns": list(service.CANONICAL_RAW_FEATURE_TABLE_COLUMNS),
            "final_feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        }
        write_new_json(root / "proposal_config.json", proposal_config)
        write_new_json(
            root / "proposal_summary.json",
            {
                "schema": "compag-curation-canonical-proposals/v1",
                "status": "PASS",
                "profile": profile,
                "sam2_execution_points_per_batch": microbatch,
                "tile_count": len(labels),
                "proposal_count": len(labels),
                "raw_feature_row_count": len(ordered_rows),
                "raw_rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
                "amg_input_count": len(labels),
                "empty_mask_count": 0,
                "merge_iou_suppressed_count": 0,
                "merge_iou_threshold": 0.75,
                "post_dedup_cap_count": 0,
                "full_masks_retained_losslessly": True,
                "mask_encoding": service.MASK_ENCODING,
                "proposals_sha256": sha256_file(root / "proposals.csv"),
                "raw_features_sha256": sha256_file(root / "features.csv"),
                "proposal_config_sha256": sha256_file(root / "proposal_config.json"),
                "overlays": None,
            },
        )
        self._seal_stage20(root)
        return root, labels

    def _bundle(
        self,
        *,
        profile: str = CANONICAL_GPU_PROFILE,
        provenance_overrides: dict[str, object] | None = None,
        output_name: str = "source_bundle_gpu",
    ) -> Path:
        state = _state(0)
        configs = _canonical_config_contracts(
            profile=profile,
            asset_profile=self.asset_profile,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        payloads = serialize_canonical_feature_state(state)
        output = self.root / output_name
        provenance = {
            "reviewed_sha256": "a" * 64,
            "split_manifest_sha256": "b" * 64,
            "raw_features_sha256": "c" * 64,
            "feature_state_fit_group_sha256": "d" * 64,
            "feature_state_fit_rows_sha256": "e" * 64,
            "feature_state_fit_scope": "OUTER_TRAIN_ROWS_BEFORE_CV",
            "validation_feature_scale": 1.0,
            "cv_score_interpretation": "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA",
            "training_row_count": state.training_row_count,
            "positive_training_row_count": state.positive_row_count,
            "fixed_threshold": 0.5,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "ubj_parity_status": "PASS",
            "paper_result_reproduction": "NOT_CLAIMED",
        }
        provenance.update(profile=profile, device="cuda")
        if provenance_overrides is not None:
            provenance.update(provenance_overrides)
        write_model_bundle_v2(
            BundleV2WriteRequest(
                output=output,
                profile=profile,
                classifier_ubj=portable_classifier_bytes(_classifier()),
                feature_order=CANONICAL_FEATURE_ORDER,
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
                imputer_statistics={name: 0.0 for name in CANONICAL_FEATURE_ORDER},
                prototype_npy=payloads.prototype_npy,
                pca_mean_npy=payloads.pca_mean_npy,
                pca_components_npy=payloads.pca_components_npy,
                pca_explained_variance=(1.0,) * 32,
                normalized_config=configs["normalized"],
                preprocessing=configs["preprocessing"],
                proposal_config=configs["proposal"],
                feature_config=configs["features"],
                training_config=configs["training"],
                compatibility={
                    "classifier_format": "XGBOOST_UBJ",
                    "no_pickle_or_joblib": True,
                    "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                    "fixed_threshold": 0.5,
                },
                provenance=provenance,
                sam2_config=self.assets / "hiera.yaml",
                sam2_checkpoint=self.assets / "hiera.pt",
                sam2_license=self.assets / "sam-license.txt",
                resnet50_weights=self.assets / "resnet50.pth",
                resnet50_license=self.assets / "resnet-license.txt",
            )
        )
        return output

    def _stage60(
        self,
        bundle: Path,
        groups: list[str],
        *,
        stage_name: str = "stage60",
        candidate_prefix: str = "",
        zero_candidate_group: str | None = None,
    ) -> tuple[Path, dict[str, int]]:
        stage = self.root / stage_name
        stage.mkdir()
        verified_bundle = verify_model_bundle(bundle)
        bundle_sha = verified_bundle.bundle_sha256
        profile = verified_bundle.profile
        device = "cuda" if profile == CANONICAL_GPU_PROFILE else "cpu"
        candidates: list[tuple[dict[str, str], dict[str, object], int]] = []
        labels: dict[str, int] = {}
        for index, group in enumerate(groups, start=1):
            name = f"{candidate_prefix}candidate-{index}"
            label = index % 2
            proposal_id = _sha(f"proposal:{name}")
            image_name = f"{group}__{name}.jpg"
            image_sha = _sha(f"image-bytes:{name}")
            image_id = hashlib.sha256(
                b"compag-image-v1\0"
                + image_name.encode("utf-8")
                + b"\0"
                + group.encode("ascii")
                + b"\0"
                + image_sha.encode("ascii")
            ).hexdigest()
            probability = 0.5 + index / 100.0
            detection_id = hashlib.sha256(
                b"compag-canonical-detection-v1\0"
                + proposal_id.encode("ascii")
                + b"\0"
                + b"1"
            ).hexdigest()
            prediction = {
                "id": "",
                "detection_id": detection_id,
                "proposal_id": proposal_id,
                "proposal_sha256": proposal_id,
                "image": image_name,
                "full_image": image_name,
                "image_id": image_id,
                "image_sha256": image_sha,
                "group_id": group,
                "tile_name": f"{name}_y00000x00000.jpg",
                "tile_sha256": _sha(f"tile:{name}"),
                "scale": "1",
                "mask_sha256": _fixture_mask_sha256(),
                "x": "2",
                "y": "3",
                "w": "10",
                "h": "11",
                "bbox_x1": "2",
                "bbox_y1": "3",
                "bbox_x2": "12",
                "bbox_y2": "14",
                "orig_w": "64",
                "orig_h": "64",
                "poly": "[2,3,12,3,12,14,2,14]",
                "xgb_p": service._format_float(probability),
                "probability": service._format_float(probability),
                "prediction": "1",
                "kept": "1",
            }
            labels[proposal_id] = label
            raw = _raw_row(
                name=name,
                group=group,
                label=label,
                proposal_index=100 + index,
                scale=1.0,
            )
            # Bind the raw fixture to the canonical inference image identity.
            raw.update(
                {
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "image_name": image_name,
                }
            )
            inventory = {
                "image_name": image_name,
                "image_id": image_id,
                "image_sha256": image_sha,
                "group_id": group,
                "size_bytes": 100 + index,
                "width": 64,
                "height": 64,
                "tile_count": 1,
                "candidate_rows": 1,
            }
            candidates.append((prediction, inventory, index))
        predictions = [
            item[0]
            for item in sorted(
                candidates,
                key=lambda item: (item[0]["image_id"], item[0]["proposal_id"]),
            )
        ]
        for index, row in enumerate(predictions, start=1):
            row["id"] = str(index)
        service._write_csv_exact(
            stage / "predictions.csv",
            service.CANONICAL_INFERENCE_COLUMNS,
            predictions,
        )
        raw_rows: list[dict[str, object]] = []
        by_name = {item[0]["image"]: item for item in candidates}
        for prediction in predictions:
            _prediction, _inventory, index = by_name[prediction["image"]]
            label = labels[prediction["proposal_id"]]
            for scale in CANONICAL_FEATURE_CROP_SCALES:
                row = _raw_row(
                    name=f"{candidate_prefix}candidate-{index}",
                    group=prediction["group_id"],
                    label=label,
                    proposal_index=100 + index,
                    scale=scale,
                )
                row.update(
                    {
                        "image_id": prediction["image_id"],
                        "image_sha256": prediction["image_sha256"],
                        "image_name": prediction["image"],
                    }
                )
                raw_rows.append(row)
        archive = adapter.write_canonical_active_learning_raw_feature_archive(
            raw_rows,
            stage / "predictions.csv",
            stage / "raw_features.csv",
            bundle_sha256=bundle_sha,
            profile=profile,
        )
        inventory = [item[1] for item in candidates]
        if zero_candidate_group is not None:
            empty_name = f"{zero_candidate_group}__hidden-empty.jpg"
            empty_sha256 = _sha("image-bytes:hidden-empty")
            empty_image_id = hashlib.sha256(
                b"compag-image-v1\0"
                + empty_name.encode("utf-8")
                + b"\0"
                + zero_candidate_group.encode("ascii")
                + b"\0"
                + empty_sha256.encode("ascii")
            ).hexdigest()
            inventory.append(
                {
                    "image_name": empty_name,
                    "image_id": empty_image_id,
                    "image_sha256": empty_sha256,
                    "group_id": zero_candidate_group,
                    "size_bytes": 99,
                    "width": 64,
                    "height": 64,
                    "tile_count": 1,
                    "candidate_rows": 0,
                }
            )
        inventory.sort(key=lambda item: str(item["image_name"]))
        result = {
            "schema": "compag-curation-canonical-inference/v1",
            "status": "PASS",
            "profile": profile,
            "device": device,
            "bundle_sha256": bundle_sha,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "threshold": CANONICAL_DECISION_THRESHOLD,
            "threshold_method": "FIXED_CANONICAL_METHOD",
            "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
            "sam2_execution_points_per_batch": CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            "image_count": len(inventory),
            "tile_count": len(inventory),
            "proposal_count": len(predictions),
            "prediction_rows": len(predictions),
            "positive_rows": len(predictions),
            "kept_rows": len(predictions),
            "predictions_sha256": sha256_file(stage / "predictions.csv"),
            "raw_feature_archive": archive,
            "input_inventory": inventory,
            "paper_result_reproduction": "NOT_CLAIMED",
        }
        write_new_json(stage / "inference_result.json", result)
        adapter.read_canonical_active_learning_inference(stage)
        return stage, labels

    def _initial_operations(self) -> tuple[Path, Path, Path]:
        export = self.root / "initial_export_operation"
        result, code = facade.initial_export(self.stage20, export)
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        reviewed = self.root / "initial_reviewed.csv"
        _reviewed_copy(
            export / "payload/initial_export/review_request.csv",
            reviewed,
            self.initial_labels,
        )
        completion = self.root / "initial_resume_operation"
        result, code = facade.initial_resume(export, reviewed, completion)
        self.assertEqual((result["status"], code), ("PASS", 0))
        return export, reviewed, completion

    def _assert_sealed(self, root: Path) -> None:
        status = json.loads((root / "FINAL_STATUS.json").read_text(encoding="ascii"))
        manifest = json.loads((root / "OUTPUT_MANIFEST.json").read_text(encoding="ascii"))
        rows = manifest_rows(root, excluded={"OUTPUT_MANIFEST.json"})
        self.assertEqual(manifest["members"], rows)
        self.assertEqual(manifest["members_sha256"], compact_json_sha256(rows))
        self.assertNotIn("OUTPUT_MANIFEST.json", {row["path"] for row in rows})
        self.assertEqual(status["status"], manifest["status"])
        for name in ("command_record.json", "input_manifest.json", "environment.json"):
            payload = (root / name).read_bytes()
            self.assertNotIn(str(self.root).encode(), payload)
            self.assertNotIn(b"/home/", payload)
        inputs = json.loads((root / "input_manifest.json").read_text(encoding="ascii"))
        environment = json.loads((root / "environment.json").read_text(encoding="ascii"))
        for record in [*inputs["inputs"], *environment["artifacts"]]:
            self.assertEqual(
                set(record),
                {"role", "basename", "sha256", "size_bytes"},
            )

    def _reseal_outer_manifest(self, root: Path) -> None:
        path = root / "OUTPUT_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="ascii"))
        rows = manifest_rows(root, excluded={"OUTPUT_MANIFEST.json"})
        manifest["members"] = rows
        manifest["members_sha256"] = compact_json_sha256(rows)
        path.write_bytes(canonical_json_bytes(manifest))

    def test_initial_operations_are_sealed_redacted_no_clobber_and_tamper_evident(self) -> None:
        export, reviewed, completion = self._initial_operations()
        self._assert_sealed(export)
        self._assert_sealed(completion)
        self.assertEqual(os.stat(export).st_mode & 0o777, 0o700)
        with self.assertRaisesRegex(PublicIOError, "already exists"):
            facade.initial_export(self.stage20, export)
        with self.assertRaisesRegex(PublicIOError, "already exists"):
            facade.initial_resume(export, reviewed, completion)
        collision_id = uuid.UUID("00000000-0000-4000-8000-000000000001")
        collision_output = self.root / "collision_output"
        collision = self.root / (
            f".{collision_output.name}.canonical-active-learning.{collision_id}"
        )
        collision.mkdir(mode=0o700)
        (collision / "foreign.txt").write_text("foreign\n", encoding="ascii")
        with mock.patch.object(facade.uuid, "uuid4", return_value=collision_id):
            with self.assertRaises(FileExistsError):
                facade.initial_export(self.stage20, collision_output)
        self.assertEqual(
            {path.name for path in collision.iterdir()},
            {"foreign.txt"},
        )
        command = export / "command_record.json"
        command.write_bytes(command.read_bytes() + b" ")
        with self.assertRaises(PublicIOError):
            facade.initial_resume(export, reviewed, self.root / "tampered_resume")
        self.assertFalse((self.root / "tampered_resume").exists())

        forged_export = self.root / "forged_environment_export"
        facade.initial_export(self.stage20, forged_export)
        environment_path = forged_export / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="ascii"))
        environment["artifacts"][0]["sha256"] = "0" * 64
        environment_path.write_bytes(canonical_json_bytes(environment))
        self._reseal_outer_manifest(forged_export)
        with self.assertRaisesRegex(PublicIOError, "evidence roles"):
            facade.initial_resume(
                forged_export,
                reviewed,
                self.root / "forged_environment_resume",
            )
            self.assertFalse((self.root / "forged_environment_resume").exists())

    def test_full_image_stage20_can_enter_generic_initial_labeling(self) -> None:
        stage20, labels = self._stage20(
            profile=FULL_IMAGE_GPU_PROFILE,
            stage_name="full_image_stage20",
        )
        output = self.root / "full_image_initial_export"
        result, code = facade.initial_export(stage20, output)
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        request = output / "payload/initial_export/review_request.csv"
        with request.open("r", encoding="ascii", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(labels))
        pool_result = json.loads(
            (
                output / "payload/initial_pool/pool_result.json"
            ).read_text(encoding="ascii")
        )
        self.assertEqual(pool_result["profile"], FULL_IMAGE_GPU_PROFILE)

    def test_filename_derived_group_and_image_identity_are_required(self) -> None:
        forged = self.root / "forged_stage20"
        shutil.copytree(self.stage20, forged)
        (forged / "_SUCCESS.json").unlink()
        features = forged / "features.csv"
        with features.open("r", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
        proposal_id = rows[0]["proposal_id"]
        for row in rows:
            if row["proposal_id"] != proposal_id:
                continue
            claimed_group = "forged"
            row["group_id"] = claimed_group
            row["image_id"] = hashlib.sha256(
                b"compag-image-v1\0"
                + row["image_name"].encode("utf-8")
                + b"\0"
                + claimed_group.encode("ascii")
                + b"\0"
                + row["image_sha256"].encode("ascii")
            ).hexdigest()
        features.unlink()
        service._write_csv_exact(features, columns, rows)
        summary_path = forged / "proposal_summary.json"
        summary = json.loads(summary_path.read_text(encoding="ascii"))
        summary["raw_features_sha256"] = sha256_file(features)
        summary_path.unlink()
        write_new_json(summary_path, summary)
        self._seal_stage20(forged)
        with self.assertRaisesRegex(PublicIOError, "filename-derived group"):
            facade.initial_export(
                forged,
                self.root / "forged_identity_export",
            )
        self.assertFalse((self.root / "forged_identity_export").exists())

    def test_resealed_stage20_semantic_drift_fails_once_without_publication(self) -> None:
        cases = (
            ("microbatch", "summary, hashes, or microbatch"),
            ("proposal_config", "proposal configuration differs"),
        )
        for kind, message in cases:
            with self.subTest(kind=kind):
                forged = self.root / f"stage20_{kind}_drift"
                shutil.copytree(self.stage20, forged)
                (forged / "_SUCCESS.json").unlink()
                summary_path = forged / "proposal_summary.json"
                summary = json.loads(summary_path.read_text(encoding="ascii"))
                if kind == "microbatch":
                    summary["sam2_execution_points_per_batch"] += 1
                else:
                    config_path = forged / "proposal_config.json"
                    config = json.loads(config_path.read_text(encoding="ascii"))
                    config["masked_crop_padding"] = 0.2
                    config_path.unlink()
                    write_new_json(config_path, config)
                    summary["proposal_config_sha256"] = sha256_file(config_path)
                summary_path.unlink()
                write_new_json(summary_path, summary)
                self._seal_stage20(forged)

                output = self.root / f"rejected_{kind}_export"
                with self.assertRaisesRegex(PublicIOError, message):
                    facade.initial_export(forged, output)
                self.assertFalse(output.exists())
                failures = list(
                    self.root.glob(
                        f".{output.name}.canonical-active-learning.*"
                    )
                )
                self.assertEqual(len(failures), 1)
                self.assertTrue((failures[0] / "_FAILURE.json").is_file())

    def test_full_initial_lineage_seeds_a_strict_gpu_first_round(self) -> None:
        _export, reviewed, initial_completion = self._initial_operations()
        initial_record = json.loads(
            (
                initial_completion
                / "payload/initial_completion/initial_completion.json"
            ).read_text(encoding="ascii")
        )
        self.assertEqual(initial_record["profile"], CANONICAL_GPU_PROFILE)

        gpu_split_root = self.root / "gpu_split"
        gpu_split_root.mkdir()
        service.build_canonical_group_split(
            reviewed,
            gpu_split_root,
            profile=CANONICAL_GPU_PROFILE,
        )
        gpu_split = gpu_split_root / "split_manifest.json"
        gpu_split_value = json.loads(gpu_split.read_text(encoding="ascii"))

        bundle = self._bundle(profile=CANONICAL_GPU_PROFILE)
        stage60, round_labels = self._stage60(
            bundle,
            list(gpu_split_value["train_groups"][:2]),
            stage_name="stage60_gpu_lineage",
        )
        decision_log = (
            initial_completion
            / "payload/initial_completion/decision_log.csv"
        )
        selection = self.root / "gpu_round_selection"
        result, code = facade.begin_round(
            stage60,
            bundle,
            gpu_split_root / "split_manifest.json",
            decision_log,
            1,
            initial_completion,
            selection,
        )
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        round_manifest = json.loads(
            (
                selection / "payload/selection/round_manifest.json"
            ).read_text(encoding="ascii")
        )
        pool_result = json.loads(
            (
                selection / "payload/round_pool/pool_result.json"
            ).read_text(encoding="ascii")
        )
        self.assertEqual(round_manifest["profile"], CANONICAL_GPU_PROFILE)
        self.assertEqual(pool_result["profile"], CANONICAL_GPU_PROFILE)

        round_reviewed = self.root / "gpu_round_reviewed.csv"
        _reviewed_copy(
            selection / "payload/selection/review_request.csv",
            round_reviewed,
            round_labels,
        )
        observed: dict[str, object] = {}

        def gpu_trainer(
            feature_rows: object,
            labels: object,
            groups: object,
            actions: object,
            weights: object,
            *,
            scales: object,
            split: object,
            config: object,
        ) -> object:
            observed["training_device"] = config.device
            observed["training_host_threads"] = config.thread_count
            return SimpleNamespace(
                classifier=_classifier(),
                imputer_statistics=(0.0,) * len(CANONICAL_FEATURE_ORDER),
                feature_order=CANONICAL_FEATURE_ORDER,
                fixed_threshold=CANONICAL_DECISION_THRESHOLD,
                split=split,
            )

        def gpu_parity(*_args: object, **kwargs: object) -> dict[str, object]:
            observed["parity_device"] = kwargs.get("device")
            return {"status": "PASS", "device": kwargs.get("device")}

        def gpu_probabilities(_predictor: object, rows: object) -> list[float]:
            observed["rescore_device"] = "cuda"
            return [0.5] * len(rows)

        completed = self.root / "gpu_round_completion"
        with mock.patch.object(
            facade,
            "revalidate_science_gpu_dependencies",
            side_effect=lambda _device, captured: dict(captured),
        ) as gpu_revalidation, mock.patch.object(
            adapter,
            "train_canonical_xgb",
            side_effect=gpu_trainer,
        ), mock.patch.object(
            adapter,
            "verify_probability_parity",
            side_effect=gpu_parity,
        ), mock.patch(
            "compag_curation.canonical.serialization.load_portable_predictor",
            return_value=object(),
        ) as predictor_loader, mock.patch(
            "compag_curation.canonical.serialization.predict_portable_probabilities",
            side_effect=gpu_probabilities,
        ):
            result, code = facade.resume_round(
                selection,
                stage60,
                self.stage20 / "features.csv",
                bundle,
                gpu_split_root / "split_manifest.json",
                decision_log,
                round_reviewed,
                1,
                initial_completion,
                completed,
            )
        self.assertEqual((result["status"], code), ("PASS", 0))
        gpu_revalidation.assert_called_once_with("cuda", self.science_dependencies)
        self.assertEqual(
            observed,
            {
                "training_device": "cuda",
                "training_host_threads": 1,
                "parity_device": "cuda",
                "rescore_device": "cuda",
            },
        )
        predictor_loader.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            device="cuda",
        )
        completion_record = json.loads(
            (
                completed / "payload/round_completion/round_completion.json"
            ).read_text(encoding="ascii")
        )
        retrain_record = json.loads(
            (
                completed / "payload/round_completion/retrain_request.json"
            ).read_text(encoding="ascii")
        )
        self.assertEqual(completion_record["profile"], CANONICAL_GPU_PROFILE)
        self.assertEqual(
            (retrain_record["schema"], retrain_record["profile"], retrain_record["device"]),
            (
                "compag-curation-canonical-active-learning-retrain-request/v2",
                CANONICAL_GPU_PROFILE,
                "cuda",
            ),
        )

    def test_lite_initial_lineage_cannot_seed_a_full_round(self) -> None:
        lite_stage20, lite_labels = self._stage20(
            profile=EFFICIENT_GPU_PROFILE,
            stage_name="stage20_lite",
        )
        export = self.root / "lite_initial_export"
        result, code = facade.initial_export(lite_stage20, export)
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        reviewed = self.root / "lite_initial_reviewed.csv"
        _reviewed_copy(
            export / "payload/initial_export/review_request.csv",
            reviewed,
            lite_labels,
        )
        initial_completion = self.root / "lite_initial_completion"
        result, code = facade.initial_resume(export, reviewed, initial_completion)
        self.assertEqual((result["status"], code), ("PASS", 0))
        initial_record = json.loads(
            (
                initial_completion
                / "payload/initial_completion/initial_completion.json"
            ).read_text(encoding="ascii")
        )
        self.assertEqual(initial_record["profile"], EFFICIENT_GPU_PROFILE)

        full_split_root = self.root / "full_split_from_lite_reviews"
        full_split_root.mkdir()
        service.build_canonical_group_split(
            reviewed,
            full_split_root,
            profile=CANONICAL_GPU_PROFILE,
        )
        full_split = full_split_root / "split_manifest.json"
        split_value = json.loads(full_split.read_text(encoding="ascii"))
        full_bundle = self._bundle(profile=CANONICAL_GPU_PROFILE)
        full_stage60, _labels = self._stage60(
            full_bundle,
            list(split_value["train_groups"][:2]),
            stage_name="stage60_full_with_lite_genesis",
        )
        rejected = self.root / "full_round_with_lite_genesis"
        with self.assertRaisesRegex(PublicIOError, "initial-labeling profile"):
            facade.begin_round(
                full_stage60,
                full_bundle,
                full_split,
                initial_completion / "payload/initial_completion/decision_log.csv",
                1,
                initial_completion,
                rejected,
            )
        self.assertFalse(rejected.exists())

    def test_two_sequential_image_operations_need_no_prior_image_reinference(self) -> None:
        _export, reviewed, initial_completion = self._initial_operations()
        split_root = self.root / "sequential_split"
        split_root.mkdir()
        service.build_canonical_group_split(
            reviewed,
            split_root,
            profile=CANONICAL_GPU_PROFILE,
        )
        split = split_root / "split_manifest.json"
        bundle = self._bundle(
            profile=CANONICAL_GPU_PROFILE,
            output_name="sequential_genesis_bundle",
            provenance_overrides={
                "reviewed_sha256": sha256_file(reviewed),
                "split_manifest_sha256": sha256_file(split),
                "raw_features_sha256": sha256_file(
                    self.stage20 / "features.csv"
                ),
            },
        )
        decision_log = (
            initial_completion
            / "payload/initial_completion/decision_log.csv"
        )

        def gpu_trainer(
            feature_rows: object,
            labels: object,
            groups: object,
            actions: object,
            weights: object,
            *,
            scales: object,
            split: object,
            config: object,
        ) -> object:
            return SimpleNamespace(
                classifier=_classifier(),
                imputer_statistics=(0.0,) * len(CANONICAL_FEATURE_ORDER),
                feature_order=CANONICAL_FEATURE_ORDER,
                fixed_threshold=CANONICAL_DECISION_THRESHOLD,
                split=split,
            )

        with mock.patch.object(
            facade,
            "revalidate_science_gpu_dependencies",
            side_effect=lambda _device, captured: dict(captured),
        ), mock.patch.object(
            adapter,
            "train_canonical_xgb",
            side_effect=gpu_trainer,
        ), mock.patch.object(
            adapter,
            "verify_probability_parity",
            return_value={"status": "PASS", "device": "cuda"},
        ), mock.patch(
            "compag_curation.canonical.serialization.load_portable_predictor",
            return_value=object(),
        ), mock.patch(
            "compag_curation.canonical.serialization.predict_portable_probabilities",
            side_effect=lambda _predictor, rows: [0.5] * len(rows),
        ):
            stage60_a, labels_a = self._stage60(
                bundle,
                ["g6"],
                stage_name="sequential_stage60_a",
            )
            selection_a = self.root / "sequential_selection_a"
            paused, code = facade.begin_image_round(
                stage60_a,
                bundle,
                self.stage20 / "features.csv",
                split,
                decision_log,
                1,
                initial_completion,
                selection_a,
            )
            self.assertEqual((paused["status"], code), ("PAUSED_FOR_REVIEW", 3))
            reviewed_a = self.root / "sequential_reviewed_a.csv"
            _reviewed_copy(
                selection_a / "payload/selection/review_request.csv",
                reviewed_a,
                labels_a,
            )
            completion_a = self.root / "sequential_completion_a"
            completed, code = facade.resume_image_round(
                selection_a,
                stage60_a,
                self.stage20 / "features.csv",
                bundle,
                split,
                decision_log,
                reviewed_a,
                1,
                initial_completion,
                completion_a,
            )
            self.assertEqual((completed["status"], code), ("PASS", 0))

            round1_root = completion_a / "payload/round_completion"
            round1_bundle = round1_root / "model_bundle"
            stage60_b, labels_b = self._stage60(
                round1_bundle,
                ["g7"],
                stage_name="sequential_stage60_b",
                candidate_prefix="round2-",
            )
            inference_b = json.loads(
                (stage60_b / "inference_result.json").read_text(encoding="ascii")
            )
            self.assertEqual(inference_b["image_count"], 1)
            self.assertEqual(inference_b["input_inventory"][0]["group_id"], "g7")
            selection_b = self.root / "sequential_selection_b"
            paused, code = facade.begin_image_round(
                stage60_b,
                round1_bundle,
                self.stage20 / "features.csv",
                split,
                round1_root / "decision_log.csv",
                2,
                completion_a,
                selection_b,
            )
            self.assertEqual((paused["status"], code), ("PAUSED_FOR_REVIEW", 3))
            reviewed_b = self.root / "sequential_reviewed_b.csv"
            _reviewed_copy(
                selection_b / "payload/selection/review_request.csv",
                reviewed_b,
                labels_b,
            )
            completion_b = self.root / "sequential_completion_b"
            completed, code = facade.resume_image_round(
                selection_b,
                stage60_b,
                self.stage20 / "features.csv",
                round1_bundle,
                split,
                round1_root / "decision_log.csv",
                reviewed_b,
                2,
                completion_a,
                completion_b,
            )
            self.assertEqual((completed["status"], code), ("PASS", 0))

        round2_root = completion_b / "payload/round_completion"
        lineage = json.loads(
            (round2_root / "model_lineage.json").read_text(encoding="ascii")
        )
        archive_rows, archive_receipt = (
            facade.read_canonical_accumulated_feature_archive(
                round2_root / "accumulated_al_features.csv"
            )
        )
        self.assertEqual(lineage["model_label"], "project-r2")
        self.assertEqual(archive_receipt["proposal_count"], 2)
        self.assertEqual({row.group_id for row in archive_rows}, {"g6", "g7"})
        verified_round2 = verify_model_bundle(round2_root / "model_bundle")
        self.assertEqual(
            verified_round2.provenance["details"]["model_label"],
            "project-r2",
        )
        self._assert_sealed(completion_a)
        self._assert_sealed(completion_b)

    def test_image_round_rejects_hidden_zero_candidate_second_input(self) -> None:
        _export, reviewed, initial_completion = self._initial_operations()
        split_root = self.root / "single_image_inventory_split"
        split_root.mkdir()
        service.build_canonical_group_split(
            reviewed,
            split_root,
            profile=CANONICAL_GPU_PROFILE,
        )
        split = split_root / "split_manifest.json"
        bundle = self._bundle(
            profile=CANONICAL_GPU_PROFILE,
            output_name="single_image_inventory_bundle",
            provenance_overrides={
                "reviewed_sha256": sha256_file(reviewed),
                "split_manifest_sha256": sha256_file(split),
                "raw_features_sha256": sha256_file(
                    self.stage20 / "features.csv"
                ),
            },
        )
        stage60, _labels = self._stage60(
            bundle,
            ["g6"],
            stage_name="stage60_with_hidden_zero_candidate_image",
            zero_candidate_group="g7",
        )
        rejected = self.root / "hidden_zero_candidate_image_selection"
        with self.assertRaisesRegex(
            PublicIOError,
            "exactly one Stage-60 input image",
        ):
            facade.begin_image_round(
                stage60,
                bundle,
                self.stage20 / "features.csv",
                split,
                initial_completion
                / "payload/initial_completion/decision_log.csv",
                1,
                initial_completion,
                rejected,
            )
        self.assertFalse(rejected.exists())

    def test_image_round_resume_rejects_hidden_zero_candidate_second_input(self) -> None:
        _export, reviewed, initial_completion = self._initial_operations()
        split_root = self.root / "resume_single_image_inventory_split"
        split_root.mkdir()
        service.build_canonical_group_split(
            reviewed,
            split_root,
            profile=CANONICAL_GPU_PROFILE,
        )
        split = split_root / "split_manifest.json"
        bundle = self._bundle(
            profile=CANONICAL_GPU_PROFILE,
            output_name="resume_single_image_inventory_bundle",
            provenance_overrides={
                "reviewed_sha256": sha256_file(reviewed),
                "split_manifest_sha256": sha256_file(split),
                "raw_features_sha256": sha256_file(
                    self.stage20 / "features.csv"
                ),
            },
        )
        stage60, labels = self._stage60(
            bundle,
            ["g6"],
            stage_name="resume_stage60_before_hidden_input",
        )
        decision_log = (
            initial_completion
            / "payload/initial_completion/decision_log.csv"
        )
        selection = self.root / "resume_hidden_input_selection"
        result, code = facade.begin_image_round(
            stage60,
            bundle,
            self.stage20 / "features.csv",
            split,
            decision_log,
            1,
            initial_completion,
            selection,
        )
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        round_reviewed = self.root / "resume_hidden_input_reviewed.csv"
        _reviewed_copy(
            selection / "payload/selection/review_request.csv",
            round_reviewed,
            labels,
        )

        inference_path = stage60 / "inference_result.json"
        inference = json.loads(inference_path.read_text(encoding="ascii"))
        hidden_name = "g7__hidden-empty.jpg"
        hidden_sha256 = _sha("image-bytes:resume-hidden-empty")
        hidden_image_id = hashlib.sha256(
            b"compag-image-v1\0"
            + hidden_name.encode("utf-8")
            + b"\0g7\0"
            + hidden_sha256.encode("ascii")
        ).hexdigest()
        inference["input_inventory"].append(
            {
                "image_name": hidden_name,
                "image_id": hidden_image_id,
                "image_sha256": hidden_sha256,
                "group_id": "g7",
                "size_bytes": 99,
                "width": 64,
                "height": 64,
                "tile_count": 1,
                "candidate_rows": 0,
            }
        )
        inference["input_inventory"].sort(
            key=lambda item: str(item["image_name"])
        )
        inference["image_count"] = 2
        inference["tile_count"] = 2
        inference_path.unlink()
        write_new_json(inference_path, inference)
        adapter.read_canonical_active_learning_inference(stage60)

        current_stage60_record = facade._directory_record(
            stage60,
            "canonical_inference_root",
        )
        for record_name in ("command_record.json", "input_manifest.json"):
            record_path = selection / record_name
            record = json.loads(record_path.read_text(encoding="ascii"))
            for input_record in record["inputs"]:
                if input_record["role"] == "canonical_inference_root":
                    input_record.clear()
                    input_record.update(current_stage60_record)
            record_path.unlink()
            write_new_json(record_path, record)
        self._reseal_outer_manifest(selection)

        rejected = self.root / "resume_hidden_zero_candidate_image"
        with self.assertRaisesRegex(
            PublicIOError,
            "exactly one Stage-60 input image",
        ):
            facade.resume_image_round(
                selection,
                stage60,
                self.stage20 / "features.csv",
                bundle,
                split,
                decision_log,
                round_reviewed,
                1,
                initial_completion,
                rejected,
            )
        self.assertFalse(rejected.exists())

    def test_round_resume_uses_concrete_adapter_and_failure_never_publishes_pass(self) -> None:
        _export, reviewed, initial_completion = self._initial_operations()
        split_root = self.root / "split"
        split_root.mkdir()
        service.build_canonical_group_split(
            reviewed,
            split_root,
            profile=CANONICAL_GPU_PROFILE,
        )
        split = split_root / "split_manifest.json"
        split_value = json.loads(split.read_text(encoding="ascii"))
        bundle = self._bundle()
        stage60, round_labels = self._stage60(bundle, list(split_value["train_groups"][:2]))
        decision_log = initial_completion / "payload/initial_completion/decision_log.csv"
        selection = self.root / "round_selection_operation"
        result, code = facade.begin_round(
            stage60,
            bundle,
            split,
            decision_log,
            1,
            initial_completion,
            selection,
        )
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        archive = adapter.read_canonical_active_learning_inference(stage60)
        pool_result = json.loads(
            (selection / "payload/round_pool/pool_result.json").read_text(
                encoding="ascii"
            )
        )
        self.assertEqual(pool_result["source_sha256"], archive.raw_features_sha256)
        self.assertEqual(
            pool_result["feature_archive_sha256"],
            archive.feature_archive_sha256,
        )
        self.assertNotEqual(
            pool_result["source_sha256"],
            pool_result["feature_archive_sha256"],
        )
        self._assert_sealed(selection)
        with self.assertRaisesRegex(PublicIOError, "already exists"):
            facade.begin_round(
                stage60,
                bundle,
                split,
                decision_log,
                1,
                initial_completion,
                selection,
            )
        round_reviewed = self.root / "round_reviewed.csv"
        _reviewed_copy(
            selection / "payload/selection/review_request.csv",
            round_reviewed,
            round_labels,
        )

        failed = self.root / "failed_round_resume"
        events: list[str] = []

        def checked_preflight(device: str) -> dict[str, object]:
            self.assertEqual(device, "cuda")
            events.append("preflight")
            return self.science_dependencies

        def stopped_trainer(*_args: object, **_kwargs: object) -> object:
            events.append("trainer")
            raise RuntimeError("trainer stopped")

        with mock.patch.object(
            facade,
            "require_science_dependencies",
            side_effect=checked_preflight,
        ), mock.patch.object(
            adapter,
            "train_canonical_xgb",
            side_effect=stopped_trainer,
        ):
            with self.assertRaisesRegex(RuntimeError, "trainer stopped"):
                facade.resume_round(
                    selection,
                    stage60,
                    self.stage20 / "features.csv",
                    bundle,
                    split,
                    decision_log,
                    round_reviewed,
                    1,
                    initial_completion,
                    failed,
                )
        self.assertEqual(events, ["preflight", "trainer"])
        self.assertFalse(failed.exists())
        failures = list(self.root.glob(f".{failed.name}.canonical-active-learning.*"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(os.stat(failures[0]).st_mode & 0o777, 0o700)
        self.assertTrue((failures[0] / "_FAILURE.json").is_file())
        self.assertFalse((failures[0] / "FINAL_STATUS.json").exists())

        rejected = self.root / "runtime_rejected_resume"
        rejected_trainer = mock.Mock(side_effect=AssertionError("trainer invoked"))
        with mock.patch.object(
            facade,
            "require_science_dependencies",
            side_effect=PublicIOError("locked science runtime required"),
        ), mock.patch.object(
            adapter,
            "train_canonical_xgb",
            rejected_trainer,
        ):
            with self.assertRaisesRegex(PublicIOError, "locked science runtime"):
                facade.resume_round(
                    selection,
                    stage60,
                    self.stage20 / "features.csv",
                    bundle,
                    split,
                    decision_log,
                    round_reviewed,
                    1,
                    initial_completion,
                    rejected,
                )
        rejected_trainer.assert_not_called()
        self.assertFalse(rejected.exists())
        self.assertEqual(
            list(self.root.glob(f".{rejected.name}.canonical-active-learning.*")),
            [],
        )

        classifier = _classifier()
        trainer_calls: list[object] = []

        def fake_trainer(
            feature_rows: object,
            labels: object,
            groups: object,
            actions: object,
            weights: object,
            *,
            scales: object,
            split: object,
            config: object,
        ) -> object:
            trainer_calls.append(feature_rows)
            self.assertEqual(config.device, "cuda")
            return SimpleNamespace(
                classifier=classifier,
                imputer_statistics=(0.0,) * len(CANONICAL_FEATURE_ORDER),
                feature_order=CANONICAL_FEATURE_ORDER,
                fixed_threshold=CANONICAL_DECISION_THRESHOLD,
                split=split,
            )

        completed = self.root / "round_resume_operation"
        with mock.patch.object(
            adapter,
            "train_canonical_xgb",
            side_effect=fake_trainer,
        ), mock.patch.object(
            adapter,
            "verify_probability_parity",
            return_value={"status": "PASS", "device": "cuda"},
        ), mock.patch.object(
            facade,
            "revalidate_science_gpu_dependencies",
            side_effect=lambda _device, captured: dict(captured),
        ), mock.patch(
            "compag_curation.canonical.serialization.load_portable_predictor",
            return_value=object(),
        ), mock.patch(
            "compag_curation.canonical.serialization.predict_portable_probabilities",
            side_effect=lambda _predictor, rows: [0.5] * len(rows),
        ):
            result, code = facade.resume_round(
                selection,
                stage60,
                self.stage20 / "features.csv",
                bundle,
                split,
                decision_log,
                round_reviewed,
                1,
                initial_completion,
                completed,
            )
        self.assertEqual((result["status"], code), ("PASS", 0))
        self.assertEqual(len(trainer_calls), 1)
        self._assert_sealed(completed)
        completed_environment = json.loads(
            (completed / "environment.json").read_text(encoding="ascii")
        )
        self.assertEqual(
            [row["role"] for row in completed_environment["artifacts"]],
            ["package_implementation", "science_dependency_identity"],
        )
        for relative in (
            "payload/round_completion/decision_log.csv",
            "payload/round_completion/retrain_request.json",
            "payload/round_completion/model_bundle/bundle.json",
            "payload/round_completion/rescored_pool/pool_result.json",
            "payload/round_completion/round_completion.json",
        ):
            self.assertTrue((completed / relative).is_file(), relative)
        with self.assertRaisesRegex(PublicIOError, "already exists"):
            facade.resume_round(
                selection,
                stage60,
                self.stage20 / "features.csv",
                bundle,
                split,
                decision_log,
                round_reviewed,
                1,
                initial_completion,
                completed,
            )

        round_two_bundle = completed / "payload/round_completion/model_bundle"
        round_two_stage, round_two_labels = self._stage60(
            round_two_bundle,
            list(split_value["train_groups"][:3]),
            stage_name="stage60_round_two_cumulative",
        )
        round_two_log = completed / "payload/round_completion/decision_log.csv"
        round_two_selection = self.root / "round_two_selection_operation"
        result, code = facade.begin_round(
            round_two_stage,
            round_two_bundle,
            split,
            round_two_log,
            2,
            completed,
            round_two_selection,
        )
        self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
        with (
            round_two_selection / "payload/selection/review_request.csv"
        ).open("r", encoding="ascii", newline="") as handle:
            round_two_request = list(csv.DictReader(handle))
        self.assertEqual(len(round_two_request), 1)
        round_two_reviewed = self.root / "round_two_reviewed.csv"
        _reviewed_copy(
            round_two_selection / "payload/selection/review_request.csv",
            round_two_reviewed,
            round_two_labels,
        )
        round_two_completed = self.root / "round_two_resume_operation"
        round_two_trainer_calls: list[object] = []

        def round_two_trainer(
            feature_rows: object,
            labels: object,
            groups: object,
            actions: object,
            weights: object,
            *,
            scales: object,
            split: object,
            config: object,
        ) -> object:
            round_two_trainer_calls.append(feature_rows)
            self.assertEqual(config.device, "cuda")
            return SimpleNamespace(
                classifier=_classifier(),
                imputer_statistics=(0.0,) * len(CANONICAL_FEATURE_ORDER),
                feature_order=CANONICAL_FEATURE_ORDER,
                fixed_threshold=CANONICAL_DECISION_THRESHOLD,
                split=split,
            )

        with mock.patch.object(
            adapter,
            "train_canonical_xgb",
            side_effect=round_two_trainer,
        ), mock.patch.object(
            adapter,
            "verify_probability_parity",
            return_value={"status": "PASS", "device": "cuda"},
        ), mock.patch.object(
            facade,
            "revalidate_science_gpu_dependencies",
            side_effect=lambda _device, captured: dict(captured),
        ), mock.patch(
            "compag_curation.canonical.serialization.load_portable_predictor",
            return_value=object(),
        ), mock.patch(
            "compag_curation.canonical.serialization.predict_portable_probabilities",
            side_effect=lambda _predictor, rows: [0.5] * len(rows),
        ):
            result, code = facade.resume_round(
                round_two_selection,
                round_two_stage,
                self.stage20 / "features.csv",
                round_two_bundle,
                split,
                round_two_log,
                round_two_reviewed,
                2,
                completed,
                round_two_completed,
            )
        self.assertEqual((result["status"], result["round_number"], code), ("PASS", 2, 0))
        self.assertEqual(len(round_two_trainer_calls), 1)
        self._assert_sealed(round_two_completed)


if __name__ == "__main__":
    unittest.main()
