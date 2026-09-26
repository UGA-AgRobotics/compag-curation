"""``compag-curation only-codes ...`` command group (profile only-codes-compat-v1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import PROFILE_ID


def add_subparsers(subparsers: Any) -> None:
    oc = subparsers.add_parser(
        "only-codes",
        help="Only_codes drop-in compatibility workflow (profile only-codes-compat-v1)",
        description=("Import an existing Only_codes workspace and run its inference, review, merge, pack, "
                     "feature export and training with the reference (legacy) behaviour."),
    )
    acts = oc.add_subparsers(dest="oc_action", required=True)

    p = acts.add_parser("import", help="import an existing Only_codes workspace (originals stay read-only)")
    p.add_argument("--legacy-root", type=Path, required=True, help="the old 'Clean' directory")
    p.add_argument("--sam2-repo-root", type=Path, required=True, help="SAM2 repository with checkpoints/")
    p.add_argument("--torch-home", type=Path, required=True, help="TORCH_HOME holding hub/checkpoints/resnet50-11ad3fa6.pth")
    p.add_argument("--image", help="current round image folder, e.g. IMG_9510")
    p.add_argument("--map", action="append", default=[], metavar="OLD_PREFIX=NEW_PREFIX",
                   help="explicit absolute-path prefix mapping for paths recorded by the old code (repeatable)")
    p.add_argument("--output", type=Path, required=True, help="new project directory (must not exist)")
    p.add_argument("--no-hash-large", action="store_true", help="record sizes instead of SHA-256 for multi-GB arrays")
    p.add_argument("--model", action="append", default=[], metavar="ROUND_DIR=SHA256",
                   help="also convert this trusted round model (repeatable), e.g. r137_hybrid=<sha256>")

    p = acts.add_parser("import-model", help="convert one trusted legacy round model into a verified safe bundle")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--round-dir", required=True, help="e.g. r137_hybrid")
    p.add_argument("--trust-legacy-pickle", required=True, metavar="SHA256",
                   help="owner-confirmed SHA-256 of the .pkl (explicit trust; required)")
    p.add_argument("--model-file", help="model file name when the round dir holds several .pkl files")

    p = acts.add_parser("doctor", help="compare this environment with the reference execution environment")

    p = acts.add_parser("status", help="show the persisted contract, inventory and round state")
    p.add_argument("--project", type=Path, required=True)

    p = acts.add_parser("assets", help="check stage-specific files and hashes before a long run")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--image", help="current image, e.g. IMG_9510")
    p.add_argument("--model", help="verified model bundle round directory")

    p = acts.add_parser("guide", help="interactive local launcher for import, run, review and continuation")
    p.add_argument("--project", type=Path, help="an existing imported project")

    p = acts.add_parser("infer", help="legacy tiled inference (runner xgb_recall) for one image folder")
    p.add_argument("--project", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="image folder name under Shared/maskout_tile")
    g.add_argument("--next-after", help="use the runner rule: smallest IMG_n greater than this one")
    p.add_argument("--model", required=True, help="round dir of a verified bundle, e.g. r137_hybrid")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="explicit device (reference: cuda); cpu is a documented non-equivalent variant")
    p.add_argument("--replace", action="store_true", help="snapshot and recompute an existing project run")
    p.add_argument("--points-per-batch", type=int,
                   help="NON-EQUIVALENT execution variant (reference 512) when the reference batch cannot run")
    p.add_argument("--detector", choices=("required", "none"), default="required",
                   help="reference hybrid preset requires the YOLO detector; 'none' selects the "
                        "detector-free branch explicitly and records it as a declared variant")

    for name, helptext in (("review", "open the loopback-only legacy reviewer for an image"),
                           ("review-status", "print review progress without opening a UI")):
        p = acts.add_parser(name, help=helptext)
        p.add_argument("--project", type=Path, required=True)
        p.add_argument("--image", required=True)
        p.add_argument("--review-labels", help="explicitly import this historical review log into the project")
        if name == "review":
            p.add_argument("--port", type=int, default=0)
            p.add_argument("--no-open-browser", action="store_true")

    p = acts.add_parser("merge", help="merge the image's review log into the COCO masters (project copy)")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--confirm-review-complete", action="store_true", help="explicit human approval")

    p = acts.add_parser("splits", help="append merged original ids to TRAIN (FREEZE+APPEND)")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--image", required=True)

    p = acts.add_parser("pack", help="incremental TRAIN embeddings / prototype / PCA / fold-safe pack")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--run-mode", choices=("all", "stage1", "stage2", "stage3", "skip"), default="all")

    p = acts.add_parser("features", help="COCO -> features_train.csv (append new rows only)")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--image", required=True, help="image whose review_labels.csv supplies review tags")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    p = acts.add_parser("train", help="Cell3A+3B training of the next round model")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    p = acts.add_parser("round", help="merge -> splits -> pack -> features -> train -> infer next image")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--image", required=True, help="image whose review has just been completed")
    p.add_argument("--confirm-review-complete", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--no-inference", action="store_true")

    p = acts.add_parser("inspect-state", help="inspect (and optionally rebuild) a project overlay written before 1.9.3")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--rebuild-into", type=Path, help="rebuild the derived TRAIN scope into this NEW directory "
                                                     "(without it the command only inspects and changes nothing)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    p = acts.add_parser("ui", help="loopback web form for importing an Only_codes workspace")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--no-open-browser", action="store_true")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def dispatch(args: argparse.Namespace) -> int:
    try:
        return _dispatch(args)
    except Exception as exc:  # user-facing JSON error, non-zero exit
        report = getattr(exc, "report", None)
        if report is not None:   # input/state abort: nothing was published
            print(json.dumps({"status": "FAILED_INPUTS", "profile": PROFILE_ID, "error_type": type(exc).__name__,
                              "error": str(exc), "report": report}, indent=2, sort_keys=True, default=str),
                  file=sys.stderr)
            return 3
        print(json.dumps({"status": "ERROR", "profile": PROFILE_ID, "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    from . import workflow
    from .project import LAYOUT, import_model, import_project, next_image_name, open_project

    a = args.oc_action
    if a == "import":
        maps = []
        for item in args.map:
            if "=" not in item:
                raise ValueError("--map expects OLD_PREFIX=NEW_PREFIX")
            o, n = item.split("=", 1)
            maps.append((o, n))
        proj = import_project(legacy_root=args.legacy_root, output=args.output, sam2_repo_root=args.sam2_repo_root,
                              torch_home=args.torch_home, image=args.image, path_mappings=maps,
                              hash_large=not args.no_hash_large)
        converted = {}
        for item in args.model:
            rd, sha = item.split("=", 1)
            converted[rd] = import_model(proj, rd, trusted_sha256=sha)["verification"]
        missing = [r["input_type"] for r in proj.manifest["inventory"] if r["verification_status"] == "MISSING"]
        _print({"status": "PASS", "profile": PROFILE_ID, "project": str(proj.root),
                "contract_sha256": proj.manifest["contract_sha256"], "missing_inputs": missing,
                "models_converted": converted,
                "matrix": str(proj.root / "INPUT_COMPATIBILITY_MATRIX.csv")})
        return 0
    if a == "doctor":
        from .doctor import environment_report

        rep = environment_report()
        _print(rep)
        return 0 if rep["status"] == "MATCHED_REFERENCE_ENVIRONMENT" and not rep["blockers"] else 2
    if a == "ui":
        from .import_ui import run_import_ui

        return run_import_ui(port=args.port, open_browser=not args.no_open_browser)
    if a == "guide":
        from .guide import run_guide

        return run_guide(initial_project=args.project)
    proj = open_project(args.project)
    if a == "assets":
        from .asset_status import inspect_project_assets

        _print(inspect_project_assets(proj, image=args.image, model_round_dir=args.model))
        return 0
    if a == "import-model":
        m = import_model(proj, args.round_dir, trusted_sha256=args.trust_legacy_pickle, model_file=args.model_file)
        _print({"status": m["verification"]["status"], "round_dir": args.round_dir, "verification": m["verification"]})
        return 0 if m["verification"]["status"] == "PASS_EXACT" else 1
    if a == "status":
        man = proj.manifest
        _print({"profile": man["profile"], "contract_sha256": man["contract_sha256"], "roots": man["roots"],
                "path_mappings": man["path_mappings"], "current_image": man.get("current_image"),
                "model_bundles": {k: v["verification"]["status"] for k, v in man["model_bundles"].items()},
                "missing_inputs": [r["input_type"] for r in man["inventory"] if r["verification_status"] == "MISSING"],
                "rounds_discovered": len(man.get("rounds_discovered", [])),
                "images_discovered": [i["name"] for i in man.get("images_discovered", [])],
                "entry_point_readiness": workflow.readiness(proj)})
        return 0
    if a == "infer":
        st = proj.state
        image = args.image or next_image_name([st.original(LAYOUT["maskout_tile_root"]),
                                              st.overlay(LAYOUT["maskout_tile_root"])], args.next_after)
        variant = {"points_per_batch": args.points_per_batch} if args.points_per_batch else None
        m = workflow.infer(proj, image, model_round_dir=args.model, device=args.device, replace=args.replace,
                           execution_variant=variant, detector=args.detector)
        _print({"status": "RUN_COMPLETED" if m["run_completed"] else "RUN_INCOMPLETE", "image": image,
                "reference_preset_requested": m["reference_preset_requested"],
                "effective_configuration_match": m["effective_configuration_match"]["status"],
                "inputs_verified": m["inputs_verified"]["status"],
                "software_environment_match": m["software_environment_match"].get("software_environment_match"),
                "hardware_match": m["software_environment_match"].get("hardware_match"),
                "execution_variant": m["execution_variant"],
                "output_parity_verification": m["output_parity_verification"]["status"],
                "run_root": str(st.overlay(f"{LAYOUT['maskout_tile_root']}/{image}")),
                "seconds": m.get("seconds"), "outputs": len(m.get("outputs", {}))})
        return 0
    if a in ("review", "review-status"):
        sess = workflow.open_review_session(proj, args.image, review_labels_source=args.review_labels)
        if a == "review-status":
            n = sum(len(sess.items(b)) for b in sess.bases)
            _print({"image": args.image, "candidates": n, "effective_labels": len(sess.labels_eff),
                    "seen": len(sess.seen), "review_csv": str(sess.review_csv)})
            return 0
        from .review_web import run_review_web

        return run_review_web(sess, port=args.port, open_browser=not args.no_open_browser,
                              on_action=lambda act, key: proj.record_transition("review-action",
                                                                                {"image": args.image, "action": act, "key": key}))
    if a == "merge":
        _print(workflow.merge(proj, args.image, confirm_review_complete=args.confirm_review_complete))
        return 0
    if a == "splits":
        _print(workflow.splits(proj, args.image))
        return 0
    if a == "inspect-state":
        from .state_audit import inspect_state, rebuild_into

        rep = inspect_state(proj)
        if args.rebuild_into:
            rep["rebuild"] = rebuild_into(proj, args.rebuild_into, device=args.device)
        else:
            rep["mode"] = "INSPECTION_ONLY_NOTHING_WRITTEN"
        _print(rep)
        return 0 if rep["status"] == "CLEAN" or args.rebuild_into else 2
    if a == "pack":
        _print(workflow.pack(proj, device=args.device, run_mode=args.run_mode))
        return 0
    if a == "features":
        _print(workflow.features(proj, args.image, device=args.device))
        return 0
    if a == "train":
        _print(workflow.train(proj, device=args.device))
        return 0
    if a == "round":
        _print(workflow.round_after_review(proj, args.image, device=args.device,
                                           confirm_review_complete=args.confirm_review_complete,
                                           run_inference=not args.no_inference))
        return 0
    raise ValueError(f"unknown only-codes action {a}")
