from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from compag_curation import cli
from compag_curation.public_config import (
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
)


class DualProfileCliTests(unittest.TestCase):
    def test_friendly_aliases_resolve_only_at_cli_boundary(self) -> None:
        self.assertEqual(cli._gpu_profile_id("full"), CANONICAL_GPU_PROFILE)
        self.assertEqual(cli._gpu_profile_id("lite"), EFFICIENT_GPU_PROFILE)
        self.assertEqual(
            cli._gpu_profile_id(EFFICIENT_GPU_PROFILE), EFFICIENT_GPU_PROFILE
        )

    def test_init_accepts_lite_alias_but_passes_immutable_profile(self) -> None:
        with (
            mock.patch(
                "compag_curation.public_config.initialize_project",
                return_value={"status": "PASS"},
            ) as initialize,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                cli.main(
                    [
                        "init-project",
                        "--output",
                        "/tmp/compag-lite-test",
                        "--profile",
                        "lite",
                    ]
                ),
                0,
            )
        initialize.assert_called_once_with(
            Path("/tmp/compag-lite-test"), profile=EFFICIENT_GPU_PROFILE
        )

    def test_real_demo_passes_selected_execution_profile(self) -> None:
        with (
            mock.patch(
                "compag_curation.public_pipeline.run_demo",
                return_value=({"status": "PASS"}, 0),
            ) as run_demo,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                cli.main(
                    [
                        "demo",
                        "--profile",
                        "real",
                        "--execution-profile",
                        "lite",
                        "--output",
                        "/tmp/compag-real-lite",
                        "--asset-root",
                        "/tmp/compag-assets",
                    ]
                ),
                0,
            )
        self.assertEqual(
            run_demo.call_args.kwargs["execution_profile"], EFFICIENT_GPU_PROFILE
        )


if __name__ == "__main__":
    unittest.main()
