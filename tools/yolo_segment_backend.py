#!/usr/bin/env python3
"""Run the optional external segmentation backend without the science-GPU env.

This entry point supports the producer's Python 3.10 Ultralytics 8.4.26
environment; the main COMPAG CLI remains Python 3.12 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compag_curation.r92_yolo import precompute_boxes, verify_checkpoint_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Attest or precompute boxes from a CJ segmentation package")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("attest", "predict-boxes"):
        command = commands.add_parser(name)
        command.add_argument("--model-package", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "predict-boxes":
            command.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "attest":
        result = verify_checkpoint_record(args.checkpoint, args.sha256, args.output,
                                          model_package=args.model_package)
    else:
        result = precompute_boxes(args.project, args.checkpoint, args.sha256, args.output,
                                  model_package=args.model_package)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
