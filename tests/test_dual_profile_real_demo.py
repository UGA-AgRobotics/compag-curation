from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from compag_curation import quick_demo
from compag_curation.canonical.spec import EFFICIENT_GPU_PROFILE


class DualProfileRealDemoTests(unittest.TestCase):
    def test_lite_real_demo_selects_only_tiny_assets_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            asset_root.mkdir()
            identities: dict[str, dict[str, object]] = {}
            for asset_id, filename in quick_demo._EFFICIENT_REAL_DEMO_ASSETS:
                payload = f"fixture:{asset_id}\n".encode("ascii")
                (asset_root / filename).write_bytes(payload)
                identities[asset_id] = {
                    "status": "PASS",
                    "asset_id": asset_id,
                    "filename": filename,
                    "sha256": "a" * 64,
                    "size_bytes": len(payload),
                }

            initialized: list[str] = []

            def initialize(project: Path, *, profile: str) -> dict[str, object]:
                initialized.append(profile)
                for relative in (
                    "data/images",
                    "data/inference_images",
                    "assets",
                ):
                    (project / relative).mkdir(parents=True, exist_ok=True)
                return {"status": "PASS"}

            dependencies = {"sha256": "b" * 64}
            operations = [
                (
                    {"status": "PAUSED_FOR_REVIEW"},
                    {"return_code": 3},
                    dependencies,
                ),
                (
                    {"status": "PASS", "run_id": "fixture-run"},
                    {"return_code": 0},
                    dependencies,
                ),
            ]
            verified_ids: list[str] = []

            def verify(_root: Path, asset_id: str) -> dict[str, object]:
                verified_ids.append(asset_id)
                return identities[asset_id]

            with (
                mock.patch.object(quick_demo, "verify_asset", side_effect=verify),
                mock.patch.object(
                    quick_demo, "initialize_project", side_effect=initialize
                ),
                mock.patch.object(
                    quick_demo,
                    "_demo_cuda_visibility_identity",
                    return_value={"cuda_visible_devices": "0"},
                ),
                mock.patch.object(
                    quick_demo,
                    "_run_fresh_demo_project_operation",
                    side_effect=operations,
                ),
                mock.patch.object(quick_demo, "_reviewed_copy"),
                mock.patch.object(
                    quick_demo,
                    "_run_fresh_demo_inference",
                    return_value={
                        "status": "PASS",
                        "dependency_sha256": dependencies["sha256"],
                    },
                ),
                mock.patch.object(
                    quick_demo,
                    "_seal_demo",
                    return_value={"status": "PASS"},
                ),
            ):
                result = quick_demo.run_real_demo(
                    root / "lite-demo",
                    asset_root,
                    execution_profile=EFFICIENT_GPU_PROFILE,
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(initialized, [EFFICIENT_GPU_PROFILE])
            self.assertEqual(
                verified_ids,
                [row[0] for row in quick_demo._EFFICIENT_REAL_DEMO_ASSETS],
            )
            self.assertIn("sam2.1-hiera-tiny-checkpoint", verified_ids)
            self.assertNotIn("sam2.1-hiera-large-checkpoint", verified_ids)


if __name__ == "__main__":
    unittest.main()
