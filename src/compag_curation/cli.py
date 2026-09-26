"""Public CLI for validation and inert workflow-plan construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import COMMANDS, ConfigurationError, load_config
from .plan import build_plan
from .public_config import (
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    PUBLIC_INITIALIZATION_PROFILES,
    SUPPORTED_PROFILES,
)
from .registry import variants
from .version import __version__


_GPU_PROFILE_ALIASES = {
    "full": CANONICAL_GPU_PROFILE,
    "lite": EFFICIENT_GPU_PROFILE,
    "full-image": FULL_IMAGE_GPU_PROFILE,
}


def _gpu_profile_id(value: str) -> str:
    """Resolve friendly CLI aliases without persisting them in artifacts."""

    return _GPU_PROFILE_ALIASES.get(value, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compag-curation")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        sub = subparsers.add_parser(command, help=f"construct the {command} domain contract")
        sub.add_argument("--config", type=Path, required=False)
        sub.add_argument("--variant", choices=variants(command))
        mode = sub.add_mutually_exclusive_group(required=True)
        mode.add_argument("--validate", action="store_true", help="validate the configuration without constructing a plan")
        mode.add_argument("--plan", action="store_true", help="print the inert domain plan")
        if command == "infer":
            mode.add_argument("--execute", action="store_true", help="execute fresh inference from a verified portable bundle")
            sub.add_argument("--images", type=Path)
            sub.add_argument("--bundle", type=Path)
            sub.add_argument("--output", type=Path)
            sub.add_argument("--device", choices=("cuda",), default="cuda")
    doctor = subparsers.add_parser("doctor", help="check one documented installation profile")
    doctor.add_argument(
        "--profile",
        choices=("base", "quick", "science-cpu", "science-gpu"),
        default="base",
    )
    init = subparsers.add_parser("init-project", help="create a new no-clobber user-project skeleton")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument(
        "--profile",
        choices=(*_GPU_PROFILE_ALIASES, *PUBLIC_INITIALIZATION_PROFILES),
        default="full",
    )
    inspect_data = subparsers.add_parser("inspect-data", help="inspect project data without scientific execution")
    inspect_data.add_argument("--config", type=Path, required=True)
    public_validate = subparsers.add_parser("validate", help="validate the runnable public-project contract")
    public_validate.add_argument("--config", type=Path, required=True)
    public_plan = subparsers.add_parser("plan", help="print the inert runnable-project plan")
    public_plan.add_argument("--config", type=Path, required=True)
    assets = subparsers.add_parser("assets", help="fetch or verify explicitly registered public assets")
    asset_actions = assets.add_subparsers(dest="asset_action", required=True)
    for action in ("fetch", "verify"):
        asset = asset_actions.add_parser(action)
        asset.add_argument("--asset-root", type=Path, required=True)
        selection = asset.add_mutually_exclusive_group()
        selection.add_argument("--asset", action="append", choices=(
                "resnet50-imagenet1k-v2-weights",
                "sam2-apache-license",
                "sam2.1-hiera-large-checkpoint",
                "sam2.1-hiera-large-config",
                "sam2.1-hiera-tiny-checkpoint",
                "sam2.1-hiera-tiny-config",
                "torchvision-bsd-license",
            ))
        selection.add_argument(
            "--profile",
            dest="asset_profile",
            choices=(*_GPU_PROFILE_ALIASES, *SUPPORTED_PROFILES),
        )
    demo = subparsers.add_parser("demo", help="run the generated-data quick or real demonstration")
    demo.add_argument("--profile", choices=("quick", "real"), required=True)
    demo.add_argument(
        "--execution-profile",
        choices=tuple(_GPU_PROFILE_ALIASES),
        help=(
            "for --profile real: Lite (SAM2 Tiny), Full tiled (SAM2 Large), "
            "or experimental Full-image multiscale (SAM2 Large)"
        ),
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--asset-root", type=Path)
    public_run = subparsers.add_parser("run", help="run or resume the public project pipeline")
    public_run.add_argument("--config", type=Path, required=True)
    destination = public_run.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--resume", type=Path)
    public_run.add_argument("--review-labels", type=Path)
    public_run.add_argument("--dry-run", action="store_true")
    review_ui = subparsers.add_parser(
        "review-ui",
        help="review a sealed Stage-20 pause or active-learning round",
    )
    review_modes = review_ui.add_subparsers(dest="review_ui_mode", required=True)
    for mode_name in ("web", "desktop", "status"):
        review_mode = review_modes.add_parser(
            mode_name,
            help={
                "web": "open the loopback-only browser reviewer",
                "desktop": "open the native Ubuntu/WSLg reviewer",
                "status": "print durable review progress without opening a UI",
            }[mode_name],
        )
        source = review_mode.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--run",
            type=Path,
            help="paused Stage-20 run for initial labeling",
        )
        source.add_argument(
            "--selection-root",
            type=Path,
            help="sealed begin-round operation root for iterative review",
        )
        review_mode.add_argument(
            "--stage60-root",
            type=Path,
            help="exact Stage-60 inference root bound to --selection-root",
        )
        review_mode.add_argument(
            "--images",
            type=Path,
            help="original inference-image directory bound to Stage 60",
        )
        review_mode.add_argument("--output", type=Path, required=True)
        review_mode.add_argument(
            "--state",
            type=Path,
            help="external session directory (default: OUTPUT.review-session)",
        )
        review_mode.add_argument(
            "--target-label",
            default="Target",
            help="presentation-only name for binary label 1 (for example CJ)",
        )
        review_mode.add_argument(
            "--non-target-label",
            default="Non-target",
            help="presentation-only name for binary label 0 (for example Non-CJ)",
        )
        review_mode.add_argument(
            "--model-preset",
            choices=("compag-cj-r92",),
            help=(
                "explicit Stage-20 transfer-assist preset; valid only with "
                "--run and the canonical Full profile"
            ),
        )
        if mode_name == "web":
            review_mode.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
            review_mode.add_argument("--port", type=int, default=0)
            review_mode.add_argument("--no-open-browser", action="store_true")
    active_learning = subparsers.add_parser(
        "active-learning",
        help="run append-only canonical active-learning operations",
    )
    active_actions = active_learning.add_subparsers(
        dest="active_learning_action",
        required=True,
    )
    initial_export = active_actions.add_parser(
        "initial-export",
        help="create an unscored export-all review operation from a sealed stage-20 root",
    )
    initial_export.add_argument("--stage20-root", type=Path, required=True)
    initial_export.add_argument("--output", type=Path, required=True)
    initial_resume = active_actions.add_parser(
        "initial-resume",
        help="complete initial labeling into an immutable decision log",
    )
    initial_resume.add_argument("--initial-root", type=Path, required=True)
    initial_resume.add_argument("--reviewed", type=Path, required=True)
    initial_resume.add_argument("--output", type=Path, required=True)
    begin_round = active_actions.add_parser(
        "begin-round",
        help="select one deterministic model-driven review round",
    )
    begin_round.add_argument("--stage60-root", type=Path, required=True)
    begin_round.add_argument("--bundle", type=Path, required=True)
    begin_round.add_argument("--split", type=Path, required=True)
    begin_round.add_argument("--decision-log", type=Path, required=True)
    begin_round.add_argument("--round-number", type=int, required=True)
    begin_round.add_argument("--ancestry-root", type=Path, required=True)
    begin_round.add_argument("--output", type=Path, required=True)
    resume_round = active_actions.add_parser(
        "resume-round",
        help="import review, retrain, and rescore one sealed round",
    )
    resume_round.add_argument("--selection-root", type=Path, required=True)
    resume_round.add_argument("--stage60-root", type=Path, required=True)
    resume_round.add_argument("--stage20-features", type=Path, required=True)
    resume_round.add_argument("--source-bundle", type=Path, required=True)
    resume_round.add_argument("--split", type=Path, required=True)
    resume_round.add_argument("--decision-log", type=Path, required=True)
    resume_round.add_argument("--reviewed", type=Path, required=True)
    resume_round.add_argument("--round-number", type=int, required=True)
    resume_round.add_argument("--ancestry-root", type=Path, required=True)
    resume_round.add_argument("--output", type=Path, required=True)
    begin_image_round = active_actions.add_parser(
        "begin-image-round",
        help=(
            "select one Top-50 uncertainty batch from exactly one new image"
        ),
    )
    begin_image_round.add_argument("--stage60-root", type=Path, required=True)
    begin_image_round.add_argument("--bundle", type=Path, required=True)
    begin_image_round.add_argument("--stage20-features", type=Path, required=True)
    begin_image_round.add_argument("--genesis-split", type=Path, required=True)
    begin_image_round.add_argument("--decision-log", type=Path, required=True)
    begin_image_round.add_argument("--round-number", type=int, required=True)
    begin_image_round.add_argument("--ancestry-root", type=Path, required=True)
    begin_image_round.add_argument("--output", type=Path, required=True)
    resume_image_round = active_actions.add_parser(
        "resume-image-round",
        help=(
            "full-retrain XGBoost from genesis plus cumulative reviewed image rounds"
        ),
    )
    resume_image_round.add_argument("--selection-root", type=Path, required=True)
    resume_image_round.add_argument("--stage60-root", type=Path, required=True)
    resume_image_round.add_argument("--stage20-features", type=Path, required=True)
    resume_image_round.add_argument("--source-bundle", type=Path, required=True)
    resume_image_round.add_argument("--genesis-split", type=Path, required=True)
    resume_image_round.add_argument("--decision-log", type=Path, required=True)
    resume_image_round.add_argument("--reviewed", type=Path, required=True)
    resume_image_round.add_argument("--round-number", type=int, required=True)
    resume_image_round.add_argument("--ancestry-root", type=Path, required=True)
    resume_image_round.add_argument("--output", type=Path, required=True)
    import_transfer = active_actions.add_parser(
        "import-r92-transfer-baseline",
        help="import the exact private recovered r92 fit snapshot into a sealed transfer baseline",
    )
    import_transfer.add_argument("--source-features", type=Path, required=True)
    import_transfer.add_argument("--output", type=Path, required=True)
    infer_transfer = active_actions.add_parser(
        "infer-r92-image",
        help="propose, featurize, and score exactly one new image with the published r92 transfer state",
    )
    infer_transfer.add_argument("--config", type=Path, required=True)
    infer_transfer.add_argument("--images", type=Path, required=True)
    infer_transfer.add_argument("--output", type=Path, required=True)
    begin_transfer = active_actions.add_parser(
        "begin-transfer-image-round",
        help="select the deterministic Top-K review batch for one post-r92 image round",
    )
    begin_transfer.add_argument("--stage60-root", type=Path, required=True)
    begin_transfer.add_argument("--transfer-baseline", type=Path, required=True)
    begin_transfer.add_argument("--round-number", type=int, required=True)
    begin_transfer.add_argument("--source-bundle", type=Path)
    begin_transfer.add_argument("--decision-log", type=Path)
    begin_transfer.add_argument("--prior-completion", type=Path)
    begin_transfer.add_argument("--output", type=Path, required=True)
    resume_transfer = active_actions.add_parser(
        "resume-transfer-image-round",
        help="fresh-fit project-rN from the transfer baseline and one completed image review",
    )
    resume_transfer.add_argument("--selection-root", type=Path, required=True)
    resume_transfer.add_argument("--stage60-root", type=Path, required=True)
    resume_transfer.add_argument("--transfer-baseline", type=Path, required=True)
    resume_transfer.add_argument("--config", type=Path, required=True)
    resume_transfer.add_argument("--reviewed", type=Path, required=True)
    resume_transfer.add_argument("--round-number", type=int, required=True)
    resume_transfer.add_argument("--source-bundle", type=Path)
    resume_transfer.add_argument("--decision-log", type=Path)
    resume_transfer.add_argument("--prior-completion", type=Path)
    resume_transfer.add_argument("--output", type=Path, required=True)
    convert = subparsers.add_parser("convert-annotations", help="convert CVAT point XML into canonical JSON without modifying the input")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-artifacts", help="verify a complete self-excluding file manifest")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    migrate = subparsers.add_parser(
        "migrate-feature-csv",
        help="add one legacy feature column while preserving the original CSV",
    )
    migrate.add_argument("--path", type=Path, required=True)
    migrate.add_argument("--column", choices=("scale", "review_tag"), required=True)
    from .only_codes.cli import add_subparsers as _add_only_codes_subparsers

    _add_only_codes_subparsers(subparsers)
    r92 = subparsers.add_parser("r92", help="verified native r92 CPU table sample and local review")
    r92_actions = r92.add_subparsers(dest="r92_action", required=True)
    p = r92_actions.add_parser("verify", help="verify the separately supplied native model ZIP or directory")
    p.add_argument("--model", type=Path, required=True)
    p = r92_actions.add_parser("sample", help="compute three synthetic example scores with the real r92 model")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", choices=("paper", "legacy"), default="paper")
    p = r92_actions.add_parser("score", help="score a user-owned ordered 93-feature CSV")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", choices=("paper", "legacy"), default="paper")
    p = r92_actions.add_parser("cache", help="verify and cache a local pinned r92 model ZIP")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p = r92_actions.add_parser("download", help="opt-in HTTPS fetch from a future reviewed asset URL")
    p.add_argument("--url", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p = r92_actions.add_parser("review", help="review every scored candidate; optional shortlist is view metadata")
    p.add_argument("--scored", type=Path, required=True)
    p.add_argument("--tiles", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--selection", type=Path, help="optional image,id shortlist shown in the reviewer")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-open-browser", action="store_true")
    p = r92_actions.add_parser("expand-review", help="copy a restricted v1 review into a new full-pool state revision")
    p.add_argument("--old-state", type=Path, required=True)
    p.add_argument("--new-state", type=Path, required=True)
    p = r92_actions.add_parser("init-images", help="prepare every canonical tile and empty-label registries outside the source folder")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = r92_actions.add_parser("live-subset", help="run SAM2 Large/ResNet50 V2/r92 on the one canonical tile per image")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = r92_actions.add_parser("infer-full", help="run SAM2 Large/ResNet50 V2/fixed r92 on every prepared tile")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--yolo-mode", choices=("off", "prompt", "fusion", "both"), default="off")
    p.add_argument("--yolo-checkpoint", type=Path)
    p.add_argument("--yolo-sha256")
    p.add_argument("--yolo-boxes", type=Path, help="verified all-tile detector box receipt from optional environment")
    p.add_argument("--checkpoint-dir", type=Path, help="durable per-tile resume directory outside project/output")
    p = r92_actions.add_parser("export", help="merge completed r92 review into a new COCO output")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--orig-coco", type=Path, required=True)
    p.add_argument("--tiled-coco", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confirm-review-complete", action="store_true")
    p = r92_actions.add_parser("verify-round", help="verify every full-round stage and source/output hash ledger")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--images", type=Path, help="optional original input folder for byte-preservation check")
    p = r92_actions.add_parser("plan-review", help="show exact included and excluded candidate identities before commit")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--scope", choices=("reviewed","all","locked"), default="reviewed")
    p.add_argument("--selection", type=Path)
    p = r92_actions.add_parser("finalize-review", help="seal explicitly reviewed decisions into an immutable project training snapshot")
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="unchanged native r92 feature-state asset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selection", type=Path, help="optional CSV with exactly image,id; omitted means all scored candidates")
    p.add_argument("--scope", choices=("reviewed","all","locked"), default="reviewed")
    p.add_argument("--correction-revision", help="explicit revision ID for changed previously committed decisions")
    p.add_argument("--parent-snapshot", type=Path)
    p.add_argument("--review-origin", choices=("human", "test_fixture"), required=True)
    p.add_argument("--confirm-review-complete", action="store_true")
    p = r92_actions.add_parser("select-review", help="choose a transparent uncertainty shortlist using the bound score policy")
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=50)
    p = r92_actions.add_parser("train-xgb", help="fresh-fit a versioned project XGBoost model from a finalized snapshot")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="unchanged native r92 feature-state asset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy", choices=("small-data-operational",), required=True)
    p.add_argument("--estimators", type=int, default=100)
    p = r92_actions.add_parser("verify-project-model", help="verify and freshly load a project-owned classifier")
    p.add_argument("--bundle", type=Path, required=True)
    p = r92_actions.add_parser("export-original-coco", help="export selected reviewed full masks in original image coordinates")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = r92_actions.add_parser("prepare-yolo", help="seal complete-tile human QA into a CJ-only YOLO dataset")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = r92_actions.add_parser("train-yolo", help="run optional local YOLO optimization on a complete QA dataset")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--initial-weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=512)
    p = r92_actions.add_parser("verify-yolo-checkpoint", help="attest a retained detector in the optional YOLO environment")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--model-package", type=Path, help="required for an external CJ segmentation checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p = r92_actions.add_parser("yolo-predict-boxes", help="predict all prepared tiles in the optional YOLO environment")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--model-package", type=Path, help="required for an external CJ segmentation checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p = r92_actions.add_parser("infer-next", help="run every tile of a new image with a verified project classifier")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="unchanged native r92 feature-state asset")
    p.add_argument("--project-model", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--yolo-mode", choices=("off", "prompt", "fusion", "both"), default="off")
    p.add_argument("--yolo-checkpoint", type=Path)
    p.add_argument("--yolo-sha256")
    p.add_argument("--yolo-boxes", type=Path)
    p.add_argument("--checkpoint-dir", type=Path)
    p = r92_actions.add_parser("activate-model-set", help="verify a complete round model set and atomically select it")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--round-id", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--xgb-bundle", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent-model-set", type=Path)
    p.add_argument("--yolo-mode", choices=("off", "prompt", "fusion", "both"), default="off")
    p.add_argument("--yolo-checkpoint", type=Path)
    p.add_argument("--yolo-sha256")
    p.add_argument("--yolo-training-record", type=Path)
    p.add_argument("--yolo-attestation", type=Path)
    p.add_argument("--yolo-external", action="store_true", help="use a separately attested external YOLO model")
    p.add_argument("--retained-from-round")
    p.add_argument("--retained-model-set", type=Path)
    p = r92_actions.add_parser("verify-model-set", help="verify a selected or older model set")
    p.add_argument("--model-set", type=Path, required=True)
    p = r92_actions.add_parser("complete-xgb-round", help="seal reviewed labels, fresh-fit XGB, and activate OFF model set")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--round-id", required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--selection", type=Path, help="only with --scope locked")
    p.add_argument("--scope", choices=("reviewed","all","locked"), default="reviewed")
    p.add_argument("--correction-revision", help="explicit revision ID for changed previously committed decisions")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent-snapshot", type=Path)
    p.add_argument("--parent-model-set", type=Path)
    p.add_argument("--estimators", type=int, default=100)
    p.add_argument("--review-origin", choices=("human", "test_fixture"), default="human")
    p = r92_actions.add_parser("infer-model-set", help="run next full-card inference with an explicit verified model set")
    p.add_argument("--model-set", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--yolo-boxes", type=Path)
    p.add_argument("--checkpoint-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "only-codes":
        from .only_codes.cli import dispatch as _only_codes_dispatch

        return _only_codes_dispatch(args)
    if args.command == "r92":
        try:
            from . import r92_sample

            threshold = r92_sample.PAPER_THRESHOLD if getattr(args, "threshold", "paper") == "paper" else r92_sample.LEGACY_THRESHOLD
            if args.r92_action == "verify":
                r92_sample._payloads(args.model)
                result = {"status":"PASS","model_sha256":r92_sample.CLASSIFIER_SHA256,"archive_sha256":r92_sample.ARCHIVE_SHA256}
            elif args.r92_action == "sample":
                result = r92_sample.synthetic_sample(args.model,args.output,threshold)
            elif args.r92_action == "score":
                result = r92_sample.score_csv(args.model,args.input,args.output,threshold)
            elif args.r92_action == "cache":
                result = {"status":"PASS","cached":str(r92_sample.cache_local(args.model,args.cache))}
            elif args.r92_action == "download":
                result = {"status":"PASS","cached":str(r92_sample.download(args.url,args.cache))}
            elif args.r92_action == "review":
                from .r92_review import review

                return review(args.scored,args.tiles,args.state,selection=args.selection,
                              port=args.port,open_browser=not args.no_open_browser)
            elif args.r92_action == "expand-review":
                from .r92_review import expand_session
                result = expand_session(args.old_state,args.new_state)
            elif args.r92_action == "init-images":
                from .r92_images import init_images

                result = init_images(args.images,args.model,args.output)
            elif args.r92_action == "live-subset":
                from .r92_live import run_subset

                result = run_subset(args.project,args.model,args.asset_root,args.output)
            elif args.r92_action == "infer-full":
                from .r92_live import run_full

                result = run_full(args.project,args.model,args.asset_root,args.output,
                    yolo_mode=args.yolo_mode,yolo_checkpoint=args.yolo_checkpoint,
                    yolo_sha256=args.yolo_sha256,yolo_boxes=args.yolo_boxes,
                    checkpoint_dir=args.checkpoint_dir)
            elif args.r92_action == "verify-round":
                from .r92_round import verify_round

                result = verify_round(args.project,args.inference,args.export,args.images)
            elif args.r92_action == "finalize-review":
                from .r92_project_model import finalize_review
                result = finalize_review(args.state,args.inference,args.model,args.output,
                    selection=args.selection,parent_snapshot=args.parent_snapshot,
                    review_origin=args.review_origin,confirm_review_complete=args.confirm_review_complete,
                    scope=args.scope,correction_revision=args.correction_revision)
            elif args.r92_action == "plan-review":
                from .r92_project_model import review_plan
                result = review_plan(args.state,scope=args.scope,selection=args.selection)
            elif args.r92_action == "select-review":
                from .r92_project_model import select_review
                result = select_review(args.inference,args.output,count=args.count)
            elif args.r92_action == "train-xgb":
                from .r92_project_model import train_small_data
                result = train_small_data(args.snapshot,args.model,args.output,n_estimators=args.estimators)
            elif args.r92_action == "verify-project-model":
                from .r92_project_model import load_project_model
                predictor = load_project_model(args.bundle)
                result = {"status":"PASS","classifier_sha256":predictor.model_sha256,
                          "feature_count":len(predictor.order)}
            elif args.r92_action == "export-original-coco":
                from .r92_project_model import export_original_coco
                result = export_original_coco(args.snapshot,args.inference,args.output)
            elif args.r92_action == "prepare-yolo":
                from .r92_yolo import prepare_yolo_dataset
                result = prepare_yolo_dataset(args.project,args.qa,args.output)
            elif args.r92_action == "train-yolo":
                from .r92_yolo import train_yolo
                device = int(args.device) if str(args.device).isdigit() else args.device
                result = train_yolo(args.dataset,args.initial_weights,args.output,
                                    epochs=args.epochs,batch=args.batch,device=device,imgsz=args.imgsz)
            elif args.r92_action == "verify-yolo-checkpoint":
                from .r92_yolo import verify_checkpoint_record
                result = verify_checkpoint_record(args.checkpoint,args.sha256,args.output,
                                                  model_package=args.model_package)
            elif args.r92_action == "yolo-predict-boxes":
                from .r92_yolo import precompute_boxes
                result = precompute_boxes(args.project,args.checkpoint,args.sha256,args.output,
                                          model_package=args.model_package)
            elif args.r92_action == "infer-next":
                from .r92_live import run_full
                result = run_full(args.project,args.model,args.asset_root,args.output,
                                  project_model=args.project_model,yolo_mode=args.yolo_mode,
                                  yolo_checkpoint=args.yolo_checkpoint,yolo_sha256=args.yolo_sha256,
                                  yolo_boxes=args.yolo_boxes,checkpoint_dir=args.checkpoint_dir)
            elif args.r92_action == "activate-model-set":
                from .r92_model_set import activate_model_set
                result = activate_model_set(args.workspace,args.round_id,args.snapshot,args.xgb_bundle,
                    args.model,args.output,yolo_mode=args.yolo_mode,yolo_checkpoint=args.yolo_checkpoint,
                    yolo_sha256=args.yolo_sha256,yolo_training_record=args.yolo_training_record,
                    yolo_attestation=args.yolo_attestation,yolo_external=args.yolo_external,
                    retained_from_round=args.retained_from_round,
                    retained_model_set=args.retained_model_set,
                    parent_model_set=args.parent_model_set)
            elif args.r92_action == "verify-model-set":
                from .r92_model_set import verify_model_set
                result = verify_model_set(args.model_set)
            elif args.r92_action == "complete-xgb-round":
                from .r92_model_set import complete_xgb_round
                result = complete_xgb_round(args.workspace,args.round_id,args.state,args.inference,
                    args.model,args.output,selection=args.selection,parent_snapshot=args.parent_snapshot,
                    parent_model_set=args.parent_model_set,review_origin=args.review_origin,
                    estimators=args.estimators,scope=args.scope,correction_revision=args.correction_revision)
            elif args.r92_action == "infer-model-set":
                from .r92_model_set import infer_model_set
                result = infer_model_set(args.model_set,args.project,args.asset_root,args.output,
                                         yolo_boxes=args.yolo_boxes,checkpoint_dir=args.checkpoint_dir)
            else:
                from .r92_review import export

                result = export(args.state,args.orig_coco,args.tiled_coco,args.output,confirm_review_complete=args.confirm_review_complete)
            print(json.dumps(result,sort_keys=True,default=str))
            return 0
        except Exception as exc:
            if args.r92_action == "train-yolo":
                failed = Path(args.output).resolve().with_name(Path(args.output).name + ".YOLO_FAILED.json")
                if not failed.exists():
                    try:
                        failed.parent.mkdir(parents=True, exist_ok=True)
                        failed.write_text(json.dumps({"schema":"compag-r92-yolo-failure/v1",
                            "status":"YOLO_FAILED","output":str(Path(args.output).resolve()),
                            "error_type":type(exc).__name__,"error":str(exc)[:500]},indent=2)+"\n")
                    except OSError:
                        pass
            print(json.dumps({"status":"ERROR","error_type":type(exc).__name__,"error":str(exc)},sort_keys=True),file=sys.stderr)
            return 1
    if args.command == "verify-artifacts":
        from .provenance import ManifestError, verify_manifest

        try:
            result = verify_manifest(args.root, args.manifest, require_complete=True)
        except (ManifestError, OSError, UnicodeError, ValueError):
            parser.error("artifact manifest verification failed")
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "migrate-feature-csv":
        from .features.extraction import migrate_legacy_feature_csv

        try:
            changed = migrate_legacy_feature_csv(args.path, args.column)
        except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "backup_policy": "PRESERVE_BACKUPS",
                    "changed": changed,
                    "column": args.column,
                    "path": str(args.path),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "infer":
        execution_values = (args.images, args.bundle, args.output)
        if args.execute:
            if args.config is not None or args.variant is not None:
                parser.error("--execute cannot be combined with --config or --variant")
            if any(value is None for value in execution_values):
                parser.error("--execute requires --images, --bundle, and --output")
        elif any(value is not None for value in execution_values):
            parser.error("execution-only arguments require --execute")
    if args.command == "run":
        if args.dry_run:
            if args.output is None or args.resume is not None or args.review_labels is not None:
                parser.error("--dry-run requires --output and cannot use --resume or --review-labels")
        elif (args.output is None) == (args.resume is None):
            parser.error("run requires exactly one of --output or --resume")
        elif args.output is not None and args.review_labels is not None:
            parser.error("review labels are accepted only when resuming")
    if (
        args.command == "demo"
        and args.profile == "quick"
        and args.execution_profile is not None
    ):
        parser.error("--execution-profile is accepted only with --profile real")
    if args.command == "review-ui":
        if args.review_ui_mode == "web" and not 0 <= args.port <= 65535:
            parser.error("--port must be between 0 and 65535")
        for option, value in (
            ("--target-label", args.target_label),
            ("--non-target-label", args.non_target_label),
        ):
            if (
                value != value.strip()
                or not 1 <= len(value) <= 40
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                parser.error(
                    f"{option} must be 1-40 visible characters without surrounding whitespace"
                )
        if args.target_label.casefold() == args.non_target_label.casefold():
            parser.error("--target-label and --non-target-label must differ")
        if args.model_preset is not None and (
            args.target_label != "CJ" or args.non_target_label != "Non-CJ"
        ):
            parser.error(
                "--model-preset compag-cj-r92 fixes the labels to "
                "--target-label CJ --non-target-label Non-CJ"
            )
        if args.selection_root is not None:
            if args.stage60_root is None or args.images is None:
                parser.error(
                    "--selection-root requires both --stage60-root and --images"
                )
            if args.model_preset is not None:
                parser.error("--model-preset is accepted only with --run")
        elif args.stage60_root is not None or args.images is not None:
            parser.error(
                "--stage60-root and --images are accepted only with --selection-root"
            )
        try:
            from .review.facade import run_review_ui

            payload, rc = run_review_ui(
                args.review_ui_mode,
                run=args.run,
                selection_root=args.selection_root,
                stage60_root=args.stage60_root,
                images=args.images,
                output=args.output,
                state=args.state,
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 0),
                open_browser=not getattr(args, "no_open_browser", False),
                target_label=args.target_label,
                non_target_label=args.non_target_label,
                model_preset=args.model_preset,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return rc
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
            return 1
    public_commands = {
        "doctor", "init-project", "inspect-data", "validate", "plan", "assets",
        "demo", "run", "active-learning", "convert-annotations",
    }
    if args.command in public_commands or (args.command == "infer" and args.execute):
        try:
            from . import public_pipeline as public

            if args.command == "doctor":
                payload, rc = public.doctor(args.profile)
            elif args.command == "init-project":
                from .public_config import initialize_project

                payload, rc = initialize_project(
                    args.output,
                    profile=_gpu_profile_id(args.profile),
                ), 0
            elif args.command == "inspect-data":
                payload, rc = public.inspect_project(args.config)
            elif args.command == "validate":
                payload, rc = public.validate_project(args.config)
            elif args.command == "plan":
                payload, rc = public.plan_project(args.config)
            elif args.command == "assets" and args.asset_action == "fetch":
                from .assets import asset_ids_for_profile

                selected_assets = args.asset or (
                    asset_ids_for_profile(_gpu_profile_id(args.asset_profile))
                    if args.asset_profile
                    else None
                )
                payload, rc = public.fetch_assets(args.asset_root, selected_assets)
            elif args.command == "assets":
                from .assets import asset_ids_for_profile

                selected_assets = args.asset or (
                    asset_ids_for_profile(_gpu_profile_id(args.asset_profile))
                    if args.asset_profile
                    else None
                )
                payload, rc = public.verify_assets(args.asset_root, selected_assets)
            elif args.command == "demo":
                argv_sha256 = hashlib.sha256(
                    json.dumps(raw_argv, ensure_ascii=True, separators=(",", ":")).encode("ascii")
                ).hexdigest()
                payload, rc = public.run_demo(
                    args.profile,
                    args.output,
                    args.asset_root,
                    execution_profile=_gpu_profile_id(
                        args.execution_profile or "full"
                    ),
                    invocation={"source": "CLI", "command": "demo", "exact_argv_sha256": argv_sha256},
                )
            elif args.command == "run":
                mode = "DRY_RUN" if args.dry_run else ("RESUME" if args.resume is not None else "NEW")
                argv_sha256 = hashlib.sha256(
                    json.dumps(raw_argv, ensure_ascii=True, separators=(",", ":")).encode("ascii")
                ).hexdigest()
                payload, rc = public.run_project(
                    args.config,
                    output=args.output,
                    resume=args.resume,
                    reviewed_labels=args.review_labels,
                    dry_run=args.dry_run,
                    invocation={"source": "CLI", "command": "run", "mode": mode, "exact_argv_sha256": argv_sha256},
                )
            elif args.command == "active-learning":
                from .canonical import active_learning_facade as active

                if args.active_learning_action == "initial-export":
                    payload, rc = active.initial_export(
                        args.stage20_root,
                        args.output,
                    )
                elif args.active_learning_action == "initial-resume":
                    payload, rc = active.initial_resume(
                        args.initial_root,
                        args.reviewed,
                        args.output,
                    )
                elif args.active_learning_action == "begin-round":
                    payload, rc = active.begin_round(
                        args.stage60_root,
                        args.bundle,
                        args.split,
                        args.decision_log,
                        args.round_number,
                        args.ancestry_root,
                        args.output,
                    )
                elif args.active_learning_action == "resume-round":
                    payload, rc = active.resume_round(
                        args.selection_root,
                        args.stage60_root,
                        args.stage20_features,
                        args.source_bundle,
                        args.split,
                        args.decision_log,
                        args.reviewed,
                        args.round_number,
                        args.ancestry_root,
                        args.output,
                    )
                elif args.active_learning_action == "begin-image-round":
                    payload, rc = active.begin_image_round(
                        args.stage60_root,
                        args.bundle,
                        args.stage20_features,
                        args.genesis_split,
                        args.decision_log,
                        args.round_number,
                        args.ancestry_root,
                        args.output,
                    )
                elif args.active_learning_action == "resume-image-round":
                    payload, rc = active.resume_image_round(
                        args.selection_root,
                        args.stage60_root,
                        args.stage20_features,
                        args.source_bundle,
                        args.genesis_split,
                        args.decision_log,
                        args.reviewed,
                        args.round_number,
                        args.ancestry_root,
                        args.output,
                    )
                elif args.active_learning_action == "import-r92-transfer-baseline":
                    from .canonical.transfer_baseline import (
                        MANIFEST_BASENAME,
                        import_r92_transfer_baseline,
                    )
                    from .public_io import compact_json_sha256, manifest_rows, sha256_file

                    baseline = import_r92_transfer_baseline(
                        args.source_features,
                        args.output,
                    )
                    payload, rc = {
                        "schema": "compag-curation-r92-transfer-baseline-import/v1",
                        "status": "PASS",
                        "output": str(baseline.root),
                        "archive_sha256": compact_json_sha256(
                            manifest_rows(baseline.root)
                        ),
                        "manifest_sha256": sha256_file(
                            baseline.root / MANIFEST_BASENAME
                        ),
                        "source_sha256": baseline.source_sha256,
                        "source_size_bytes": baseline.source_size_bytes,
                        "source_groups_sha256": baseline.source_groups_sha256,
                        "source_group_count": len(baseline.source_groups),
                        "row_count": baseline.row_count,
                        "proposal_count": baseline.proposal_count,
                        "train_group_count": len(baseline.train_groups),
                        "test_group_count": len(baseline.test_groups),
                    }, 0
                elif args.active_learning_action == "infer-r92-image":
                    from .canonical.transfer_inference import infer_r92_transfer_image

                    payload, rc = infer_r92_transfer_image(
                        args.config,
                        args.images,
                        args.output,
                    )
                elif args.active_learning_action == "begin-transfer-image-round":
                    payload, rc = active.begin_transfer_image_round(
                        args.stage60_root,
                        args.transfer_baseline,
                        args.round_number,
                        args.output,
                        source_bundle=args.source_bundle,
                        decision_log=args.decision_log,
                        prior_completion=args.prior_completion,
                    )
                else:
                    payload, rc = active.resume_transfer_image_round(
                        args.selection_root,
                        args.stage60_root,
                        args.transfer_baseline,
                        args.reviewed,
                        args.round_number,
                        args.output,
                        config=args.config,
                        source_bundle=args.source_bundle,
                        decision_log=args.decision_log,
                        prior_completion=args.prior_completion,
                    )
            elif args.command == "convert-annotations":
                payload, rc = public.convert_annotations(args.input, args.output)
            else:
                argv_sha256 = hashlib.sha256(
                    json.dumps(raw_argv, ensure_ascii=True, separators=(",", ":")).encode("ascii")
                ).hexdigest()
                payload, rc = public.infer_bundle(
                    args.images,
                    args.bundle,
                    args.output,
                    device=args.device,
                    invocation={"source": "CLI", "command": "infer", "exact_argv_sha256": argv_sha256},
                )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return rc
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
            return 1
    if args.config is None:
        parser.error("--config is required for workflow plan construction")
    try:
        config = load_config(args.config)
        plan = build_plan(args.command, config, args.variant)
        if args.validate:
            print(
                json.dumps(
                    {"command": args.command, "status": "VALID", "variant": plan.variant},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        print(json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")))
        return 0
    except (
        ConfigurationError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2
