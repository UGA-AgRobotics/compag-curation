"""Small terminal launcher for the existing Only_codes workflow commands.

The guide delegates every operation to the same public CLI entry points.  It
does not modify the scientific algorithm or infer that a review is complete.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .asset_status import inspect_project_assets
from .project import open_project

_IMAGE = re.compile(r"IMG_[0-9]+\Z")


def _config_file() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "compag-curation" / "laptop-guide.json"


def _remember(project: Path) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"project": str(project.resolve())}, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _recent_project() -> Path | None:
    try:
        return Path(json.loads(_config_file().read_text(encoding="utf-8"))["project"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run(args: list[str], project: Path | None = None) -> int:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    command = [sys.executable, "-m", "compag_curation", "only-codes", *args]
    print("\nRunning:", " ".join(command), flush=True)
    result = subprocess.run(command, env=env, check=False)
    if project and project.is_dir():
        log = project / "GUIDE_COMMANDS.jsonl"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "arguments": ["only-codes", *args], "return_code": result.returncode}) + "\n")
    return int(result.returncode)


def _run_r92(args: list[str]) -> int:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    command = [sys.executable, "-m", "compag_curation", "r92", *args]
    print("\nRunning:", " ".join(command), flush=True)
    return int(subprocess.run(command, env=env, check=False).returncode)


def _ask(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _stage_ok(project: Path, image: str | None, stage: str, model: str | None = None) -> bool:
    report = inspect_project_assets(open_project(project), image=image, model_round_dir=model)
    row = report["stages"][stage]
    print(f"\n{stage}: {row['status']}")
    for item in row["items"]:
        print(f"  {item['role']}: {item['status']} — {item['path']}")
    if row["status"] != "READY_FOR_RUNTIME_PREFLIGHT":
        print("Stopped before the operation. Supply the missing or invalid inputs first.")
        return False
    return True


def _reference_environment_ready() -> bool:
    from .doctor import environment_report

    report = environment_report()
    if report["status"] == "MATCHED_REFERENCE_ENVIRONMENT" and not report["blockers"]:
        return True
    print("The recorded reference inference/training environment is unavailable here.")
    for blocker in report["blockers"]:
        print("  -", blocker)
    if report["differences"]:
        print("  - software versions differ from the recorded reference")
    print("No CUDA run or fallback has been started.")
    return False


def run_guide(*, initial_project: Path | None = None) -> int:
    project = Path(initial_project).expanduser().resolve() if initial_project else _recent_project()
    if project and not (project / "ONLY_CODES_PROJECT.json").is_file():
        print(f"Saved project is unavailable: {project}")
        project = None
    print("COMPAG laptop guide — r92 project rounds and Only_codes compatible workflow")
    print("The r92 table sample is CPU-only; full-card image inference uses the verified GPU profile.")
    while True:
        print(f"\nProject: {project or '(none selected)'}")
        print("1  Import a compatible existing or own-data workspace")
        print("2  Check project status and assets")
        print("3  Score a prepared image")
        print("4  Review candidates in the local browser")
        print("5  Merge/export reviewed labels")
        print("6  Continue full training round")
        print("7  Choose an already imported project")
        print("8  Run real r92 classifier on three synthetic feature rows")
        print("9  Score your compatible 93-feature table with r92")
        print("10 Review newly scored r92 candidates with your own tiles")
        print("11 Export a completed r92 review to new COCO files")
        print("12 Prepare every tile of new full-card images")
        print("13 Run full-card r92 inference (YOLO off)")
        print("14 Finalize a selected human review batch")
        print("15 Fresh-train a project XGBoost model")
        print("16 Verify and activate a complete model set")
        print("17 Infer the next full card with a selected model set")
        print("18 Prepare a completeness-certified YOLO dataset")
        print("19 Train the optional CJ detector")
        print("20 Export selected reviewed masks to original COCO")
        print("21 Select a raw-XGB uncertainty review batch")
        print("22 Complete reviewed XGBoost round and activate next model")
        print("23 Precompute YOLO boxes in the optional detector environment")
        print("24 Verify a retained YOLO checkpoint in the optional environment")
        print("25 Expand a restricted r92 Top-K review to a new full-pool state")
        print("26 Preview the r92 review decisions entering the next snapshot")
        print("0  Exit")
        try:
            choice = _ask("Choose")
            if choice == "0":
                return 0
            if choice == "1":
                legacy = _ask("Compatible workspace root (prepared Only_codes layout)")
                sam2 = _ask("SAM2 source/weights root")
                torch_home = _ask("TORCH_HOME for ResNet50 weights")
                output = _ask("New project directory (must not exist)")
                image = _ask("Current IMG_number folder (optional)")
                if not all((legacy, sam2, torch_home, output)):
                    print("All roots and the new project path are required.")
                    continue
                args = ["import", "--legacy-root", legacy, "--sam2-repo-root", sam2,
                        "--torch-home", torch_home, "--output", output]
                if image:
                    if not _IMAGE.fullmatch(image):
                        print("Image must have the form IMG_1234.")
                        continue
                    args.extend(["--image", image])
                if _run(args) == 0:
                    project = Path(output).expanduser().resolve()
                    _remember(project)
            elif choice == "7":
                chosen = Path(_ask("Imported project directory")).expanduser().resolve()
                open_project(chosen)
                project = chosen
                _remember(project)
            elif choice == "8":
                model = _ask("Pinned r92 model ZIP (separate download)")
                output = _ask("New synthetic sample output directory")
                if model and output:
                    _run_r92(["sample", "--model", model, "--output", output])
            elif choice == "9":
                model = _ask("Pinned r92 model ZIP")
                table = _ask("Your prepared 93-feature CSV")
                output = _ask("New score output directory")
                if model and table and output:
                    _run_r92(["score", "--model", model, "--input", table, "--output", output])
            elif choice == "10":
                scored = _ask("r92 score output detections.csv")
                tiles = _ask("Your matching image tiles directory")
                state = _ask("New or matching review state directory")
                selected = _ask("Optional Top-K recommendation CSV (blank = all candidates)")
                if scored and tiles and state:
                    args = ["review", "--scored", scored, "--tiles", tiles, "--state", state]
                    if selected:
                        args.extend(["--selection", selected])
                    _run_r92(args)
            elif choice == "11":
                state = _ask("Completed r92 review state directory")
                orig = _ask("Original-image COCO input")
                tiled = _ask("Tiled COCO input")
                output = _ask("New export directory")
                if state and orig and tiled and output:
                    print("Every candidate needs an effective human label. No label is auto-confirmed.")
                    if _ask("Type REVIEW COMPLETE to authorize export") == "REVIEW COMPLETE":
                        _run_r92(["export", "--state", state, "--orig-coco", orig,
                                  "--tiled-coco", tiled, "--output", output,
                                  "--confirm-review-complete"])
                    else:
                        print("Export not started.")
            elif choice == "12":
                images = _ask("Folder of new full-card photos")
                model = _ask("Pinned native r92 model ZIP")
                output = _ask("New prepared project directory")
                if all((images, model, output)):
                    _run_r92(["init-images", "--images", images, "--model", model, "--output", output])
            elif choice == "13":
                prepared = _ask("Prepared full-card image project")
                model = _ask("Pinned native r92 model ZIP")
                assets = _ask("Verified SAM2/ResNet asset folder")
                output = _ask("New inference output directory")
                checkpoint = _ask("Resume checkpoint directory (blank = no resume)")
                if all((prepared, model, assets, output)):
                    args = ["infer-full", "--project", prepared, "--model", model,
                            "--asset-root", assets, "--output", output]
                    if checkpoint:
                        args.extend(["--checkpoint-dir", checkpoint])
                    _run_r92(args)
            elif choice == "14":
                state = _ask("Saved human review state directory")
                inference = _ask("Matching complete inference directory")
                model = _ask("Pinned native r92 model ZIP")
                scope = _ask("Commit scope: reviewed (default), all, or locked") or "reviewed"
                selected = _ask("Locked selection CSV (only for locked scope)") if scope == "locked" else ""
                parent = _ask("Previous cumulative snapshot JSON (blank for first round)")
                output = _ask("New cumulative snapshot JSON")
                if all((state, inference, model, output)) and scope in {"reviewed","all","locked"} and _ask("Type REVIEW COMPLETE after checking the declared scope") == "REVIEW COMPLETE":
                    args = ["finalize-review", "--state", state, "--inference", inference,
                            "--model", model, "--output", output, "--review-origin", "human",
                            "--confirm-review-complete", "--scope", scope]
                    if selected:
                        args.extend(["--selection", selected])
                    if parent:
                        args.extend(["--parent-snapshot", parent])
                    _run_r92(args)
            elif choice == "15":
                snapshot = _ask("Finalized cumulative snapshot JSON")
                model = _ask("Pinned native r92 model ZIP")
                output = _ask("New versioned XGBoost model directory")
                if all((snapshot, model, output)):
                    _run_r92(["train-xgb", "--snapshot", snapshot, "--model", model,
                              "--output", output, "--policy", "small-data-operational"])
            elif choice == "16":
                workspace = _ask("Existing round workspace directory")
                round_id = _ask("New round ID (e.g. project_r1)")
                snapshot = _ask("Finalized snapshot JSON")
                xgb = _ask("New XGBoost model directory")
                model = _ask("Pinned native r92 model ZIP")
                output = _ask("New model-set directory inside workspace")
                mode = _ask("YOLO inference mode: off/prompt/fusion/both") or "off"
                args = ["activate-model-set", "--workspace", workspace, "--round-id", round_id,
                        "--snapshot", snapshot, "--xgb-bundle", xgb, "--model", model,
                        "--output", output, "--yolo-mode", mode]
                if mode != "off":
                    checkpoint = _ask("Verified one-class CJ detector checkpoint")
                    digest = _ask("Checkpoint SHA-256")
                    trained = _ask("New YOLO training record JSON (blank if retaining older detector)")
                    retained = "" if trained else _ask("Older detector round ID")
                    args.extend(["--yolo-checkpoint", checkpoint, "--yolo-sha256", digest])
                    if trained:
                        args.extend(["--yolo-training-record", trained])
                    if retained:
                        args.extend(["--retained-from-round", retained])
                        retained_set = _ask("Verified older model-set directory that owns this detector")
                        args.extend(["--retained-model-set", retained_set])
                        attested = _ask("Retained detector attestation JSON")
                        args.extend(["--yolo-attestation", attested])
                if all((workspace, round_id, snapshot, xgb, model, output)):
                    _run_r92(args)
            elif choice == "17":
                model_set = _ask("Verified model-set directory (latest or older)")
                prepared = _ask("New prepared full-card image project")
                assets = _ask("Verified SAM2/ResNet asset folder")
                output = _ask("New inference output directory")
                boxes = _ask("Verified YOLO boxes JSON (blank when model-set mode is OFF)")
                checkpoint = _ask("Resume checkpoint directory (blank = no resume)")
                if all((model_set, prepared, assets, output)):
                    args = ["infer-model-set", "--model-set", model_set, "--project", prepared,
                            "--asset-root", assets, "--output", output]
                    if boxes:
                        args.extend(["--yolo-boxes", boxes])
                    if checkpoint:
                        args.extend(["--checkpoint-dir", checkpoint])
                    _run_r92(args)
            elif choice == "18":
                prepared = _ask("Prepared image project")
                qa = _ask("Completed per-tile CJ QA JSON")
                output = _ask("New YOLO dataset directory")
                if all((prepared, qa, output)):
                    _run_r92(["prepare-yolo", "--project", prepared, "--qa", qa, "--output", output])
            elif choice == "19":
                dataset = _ask("Verified complete-tile YOLO dataset directory")
                initial = _ask("Explicit one-class CJ checkpoint or one-class YAML")
                output = _ask("New YOLO training output directory")
                if all((dataset, initial, output)):
                    _run_r92(["train-yolo", "--dataset", dataset, "--initial-weights", initial,
                              "--output", output])
            elif choice == "20":
                snapshot = _ask("Finalized review snapshot JSON")
                inference = _ask("Matching complete inference directory")
                output = _ask("New original-coordinate COCO output JSON")
                if all((snapshot, inference, output)):
                    _run_r92(["export-original-coco", "--snapshot", snapshot,
                              "--inference", inference, "--output", output])
            elif choice == "21":
                inference = _ask("Complete full-card inference directory")
                count = _ask("Number of candidates (default 50)") or "50"
                output = _ask("New selection CSV")
                if all((inference, count, output)):
                    _run_r92(["select-review", "--inference", inference,
                              "--count", count, "--output", output])
            elif choice == "22":
                workspace = _ask("Existing workspace directory")
                round_id = _ask("New round ID")
                state = _ask("Saved human review state directory")
                inference = _ask("Matching complete inference directory")
                model = _ask("Pinned native r92 model ZIP")
                scope = _ask("Commit scope: reviewed (default), all, or locked") or "reviewed"
                selection = _ask("Locked selection CSV (only for locked scope)") if scope == "locked" else ""
                parent_snapshot = _ask("Previous cumulative snapshot JSON (blank for first round)")
                parent_set = _ask("Previous verified model-set directory (blank for first round)")
                output = _ask("New round output directory inside workspace")
                if all((workspace,round_id,state,inference,model,output)) and scope in {"reviewed","all","locked"} and _ask("Type REVIEW COMPLETE") == "REVIEW COMPLETE":
                    args = ["complete-xgb-round","--workspace",workspace,"--round-id",round_id,
                            "--state",state,"--inference",inference,"--model",model,
                            "--scope",scope,"--output",output]
                    if selection:
                        args.extend(["--selection",selection])
                    if parent_snapshot:
                        args.extend(["--parent-snapshot",parent_snapshot])
                    if parent_set:
                        args.extend(["--parent-model-set",parent_set])
                    _run_r92(args)
            elif choice == "23":
                prepared = _ask("Prepared full-card image project")
                checkpoint = _ask("Verified one-class CJ checkpoint")
                digest = _ask("Checkpoint SHA-256")
                output = _ask("New all-tile YOLO boxes JSON")
                if all((prepared,checkpoint,digest,output)):
                    _run_r92(["yolo-predict-boxes","--project",prepared,"--checkpoint",checkpoint,
                              "--sha256",digest,"--output",output])
            elif choice == "24":
                checkpoint = _ask("Retained one-class CJ checkpoint")
                digest = _ask("Checkpoint SHA-256")
                output = _ask("New checkpoint attestation JSON")
                if all((checkpoint,digest,output)):
                    _run_r92(["verify-yolo-checkpoint","--checkpoint",checkpoint,
                              "--sha256",digest,"--output",output])
            elif choice == "25":
                old = _ask("Restricted r92 review state")
                new = _ask("New full-pool review state directory")
                if old and new:
                    _run_r92(["expand-review","--old-state",old,"--new-state",new])
            elif choice == "26":
                state = _ask("r92 review state")
                scope = _ask("Scope: reviewed (default), all, or locked") or "reviewed"
                selection = _ask("Locked selection CSV") if scope == "locked" else ""
                if state and scope in {"reviewed","all","locked"}:
                    args=["plan-review","--state",state,"--scope",scope]
                    if selection: args.extend(["--selection",selection])
                    _run_r92(args)
            elif project is None:
                print("Import or choose a project first.")
            elif choice == "2":
                _run(["status", "--project", str(project)], project)
                _run(["assets", "--project", str(project)], project)
            elif choice == "3":
                image = _ask("Prepared IMG_number folder")
                model = _ask("Verified bundle round (e.g. r137_hybrid)")
                if not _IMAGE.fullmatch(image) or not model:
                    print("A valid image and verified model round are required.")
                    continue
                if _stage_ok(project, image, "infer", model) and _reference_environment_ready():
                    _run(["infer", "--project", str(project), "--image", image,
                          "--model", model], project)
            elif choice == "4":
                image = _ask("Image to review")
                if not _IMAGE.fullmatch(image):
                    print("Image must have the form IMG_1234.")
                    continue
                if _stage_ok(project, image, "review"):
                    _run(["review", "--project", str(project), "--image", image], project)
            elif choice == "5":
                image = _ask("Reviewed image")
                if not _IMAGE.fullmatch(image):
                    print("Image must have the form IMG_1234.")
                    continue
                if not _stage_ok(project, image, "merge"):
                    continue
                print("This merges reviewed labels into the project's COCO export overlay.")
                print("Only proceed after human review of this exact image is complete.")
                if _ask("Type REVIEW COMPLETE to authorize this round") == "REVIEW COMPLETE":
                    _run(["merge", "--project", str(project), "--image", image,
                          "--confirm-review-complete"], project)
                else:
                    print("Merge not started.")
            elif choice == "6":
                image = _ask("Reviewed image")
                if not _IMAGE.fullmatch(image):
                    print("Image must have the form IMG_1234.")
                    continue
                if not _stage_ok(project, image, "merge") or not _stage_ok(project, image, "pack"):
                    continue
                if not _reference_environment_ready():
                    continue
                print("This starts merge, splits, pack, features, a real fit and next-image inference.")
                print("Only proceed after human review of this exact image is complete.")
                if _ask("Type REVIEW COMPLETE to authorize this round") == "REVIEW COMPLETE":
                    _run(["round", "--project", str(project), "--image", image,
                          "--confirm-review-complete"], project)
                else:
                    print("Round not started.")
            else:
                print("Choose one of the displayed numbers.")
        except (EOFError, KeyboardInterrupt):
            print("\nStopped. The project remains available for a later restart.")
            return 130
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Cannot continue: {exc}")
