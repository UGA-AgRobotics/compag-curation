#!/usr/bin/env python3
"""Execute unchanged JR04 S3 code against the current sanitized derivative.

The adapter validates four separate layers before execution:
1. archival source identity (historical pins remain historical),
2. current derivative byte integrity,
3. scientific-field projections and populations, and
4. expected saved-data numerical outputs.

It does not import a model, fit, rescore, infer from images, or modify its inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from jr04_contract_lib import (
    first_difference,
    json_dump,
    scientific_contract_snapshot,
    sha256_file,
    tree_pins,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_contract_ledger(contract: Path) -> None:
    ledger = contract / "SHA256SUMS"
    require(ledger.is_file(), "Missing contract SHA256SUMS")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = contract / name
        require(path.is_file(), f"Missing contract member: {name}")
        require(sha256_file(path) == expected, f"Contract member identity mismatch: {name}")


def validate_pin_tree(root: Path, expected: dict[str, dict[str, object]], label: str) -> None:
    actual = tree_pins(root)
    if label == "candidate":
        actual.pop("JR04_CANDIDATE_PATH_REPAIR_RECEIPT.json", None)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"Missing active derivative input ({label}): {missing[0]}")
    if unexpected:
        raise ValueError(f"Unexpected active derivative input ({label}): {unexpected[0]}")
    for rel in sorted(expected):
        require(
            actual[rel] == expected[rel],
            f"Active derivative identity mismatch ({label}): {rel}",
        )


def preflight(args: argparse.Namespace) -> dict[str, object]:
    point = args.point_inputs.resolve()
    candidate = args.candidate_inputs.resolve()
    contract = args.contract.resolve()
    runner = args.runner.resolve()
    historical_pins_path = args.historical_point_pins.resolve()
    package_invariants = args.package_invariants.resolve()
    for path, label in (
        (point, "point inputs"),
        (candidate, "candidate inputs"),
        (contract, "contract"),
    ):
        require(path.is_dir(), f"Missing {label}")
    for path, label in (
        (runner, "preserved S3 runner"),
        (historical_pins_path, "historical point pins"),
        (package_invariants, "package scientific invariants"),
    ):
        require(path.is_file(), f"Missing {label}")

    verify_contract_ledger(contract)
    active_point = load_json(contract / "POINT_DERIVATIVE_INPUT_PINS.json")
    active_candidate_manifest = load_json(contract / "JR04_CANDIDATE_ACTIVE_MANIFEST.json")
    crosswalk = load_json(contract / "JR04_DERIVATIVE_IDENTITY_CROSSWALK.json")
    science_contract = load_json(contract / "JR04_SCIENTIFIC_CONTRACT.json")
    repair_receipt = load_json(contract / "JR04_CANDIDATE_PATH_REPAIR_RECEIPT.json")

    require(crosswalk["status"] == "PASS", "Identity crosswalk status is not PASS")
    require(science_contract["status"] == "PASS", "Scientific contract status is not PASS")
    require(repair_receipt["status"] == "PASS", "Candidate path repair status is not PASS")
    require(len(active_point) == 54, "Active point pin count is not 54")
    require(len(active_candidate_manifest["files"]) == 7, "Active candidate pin count is not 7")
    crosswalk_point = {row["path"]: row for row in crosswalk["point_files"]}
    require(set(crosswalk_point) == set(active_point), "Crosswalk/active point path set differs")
    for rel, pin in active_point.items():
        require(
            crosswalk_point[rel]["active_derivative"] == pin,
            f"Crosswalk/active point identity differs: {rel}",
        )
    validate_pin_tree(point, active_point, "point")
    validate_pin_tree(candidate, active_candidate_manifest["files"], "candidate")

    require(
        sha256_file(contract / "POINT_HISTORICAL_PINS_ARCHIVAL.json")
        == crosswalk["historical_point_pins"]["sha256"]
        == sha256_file(historical_pins_path),
        "Historical point pins are not preserved byte-for-byte",
    )
    historical_pins = load_json(historical_pins_path)
    changed = [
        rel
        for rel in sorted(active_point)
        if active_point[rel]["bytes"] != historical_pins[rel]["bytes"]
        or active_point[rel]["sha256"] != historical_pins[rel]["sha256"]
    ]
    require(len(changed) == 38, f"Historical/current point identity count differs: {len(changed)}")
    require(
        crosswalk["point_files_changed_from_history"] == 38,
        "Crosswalk does not preserve 38 changed point identities",
    )

    code = science_contract["preserved_code"]
    require(sha256_file(runner) == code["s3_runner_sha256"], "Preserved S3 runner changed")
    require(
        active_point["eval_points_coverage_v3.py"]["sha256"] == code["evaluator_sha256"],
        "Evaluator identity differs from scientific contract",
    )
    require(
        sha256_file(package_invariants)
        == science_contract["package_scientific_invariants"]["sha256"],
        "Package scientific-invariants receipt changed",
    )

    current_science = scientific_contract_snapshot(candidate, point)
    difference = first_difference(
        science_contract["semantic_snapshot"], current_science, "scientific_snapshot"
    )
    require(difference is None, f"Scientific projection mismatch: {difference}")
    candidate_population = current_science["candidate_population"]
    require(candidate_population["rows"] == 10097, "Candidate population is not 10097")
    require(
        candidate_population["unique_img_folder_image_id"] == 10097,
        "Candidate scientific IDs are not 10097 unique rows",
    )
    require(
        candidate_population["relative_model_identity_join"][
            "all_table4_relative_model_ids_resolve_uniquely"
        ],
        "Candidate relative model IDs do not resolve 10/10",
    )
    require(current_science["s3"]["total_xml_points"] == 740, "XML points are not 740")
    require(
        current_science["s3"]["saved_tau050_point_rows"] == 740,
        "Saved tau=0.50 point rows are not 740",
    )
    return {
        "status": "PASS",
        "candidate_rows": 10097,
        "candidate_unique_scientific_ids": 10097,
        "candidate_table4_relative_model_id_matches": 10,
        "point_input_files": 54,
        "point_files_changed_from_historical_source": 38,
        "xml_points": 740,
        "saved_tau050_point_rows": 740,
        "preserved_s3_runner_sha256": sha256_file(runner),
        "preserved_evaluator_sha256": active_point["eval_points_coverage_v3.py"]["sha256"],
        "historical_point_pins_sha256": sha256_file(historical_pins_path),
        "active_derivative_pins_sha256": sha256_file(
            contract / "POINT_DERIVATIVE_INPUT_PINS.json"
        ),
    }


def portable_command() -> list[str]:
    return [
        "$PYTHON",
        "-B",
        "$PUBLIC_ROOT/paper_reproduction/jr04/run_s3_v1.py",
        "--inputs",
        "$REVIEWER_ROOT/replay/jr04/inputs/point",
        "--pins",
        "$REVIEWER_ROOT/replay/jr04/contracts/POINT_DERIVATIVE_INPUT_PINS.json",
        "--out",
        "$NEW_OUTPUT/s3",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-inputs", type=Path, required=True)
    parser.add_argument("--candidate-inputs", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--historical-point-pins", type=Path, required=True)
    parser.add_argument("--package-invariants", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    preflight_result = preflight(args)
    if args.preflight_only:
        print(json.dumps({"preflight": preflight_result}, indent=2, sort_keys=True))
        return
    require(args.out is not None, "--out is required unless --preflight-only is used")
    out = args.out.resolve()
    require(not out.exists(), "Refusing to overwrite adapter output")

    point_before = tree_pins(args.point_inputs.resolve())
    candidate_before = tree_pins(args.candidate_inputs.resolve())
    out.mkdir(parents=True)
    s3_out = out / "raw_s3_local"
    command = [
        sys.executable,
        "-B",
        str(args.runner.resolve()),
        "--inputs",
        str(args.point_inputs.resolve()),
        "--pins",
        str((args.contract.resolve() / "POINT_DERIVATIVE_INPUT_PINS.json")),
        "--out",
        str(s3_out),
    ]
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    started = time.time()
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=1200,
    )
    elapsed = time.time() - started
    (out / "S3_RUN_LOCAL.log").write_text(process.stdout, encoding="utf-8")
    json_dump(
        out / "S3_COMMAND_RECEIPT.json",
        {
            "schema": "jr04-sanitized-s3-command/v1",
            "portable_exact_argv": portable_command(),
            "environment": {"MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
            "return_code": process.returncode,
            "elapsed_seconds": elapsed,
            "actual_argv_not_embedded_to_avoid_workstation_path_disclosure": True,
        },
    )
    require(process.returncode == 0, f"Preserved S3 runner failed with rc={process.returncode}")

    verification = load_json(s3_out / "VERIFICATION_RESULT.json")
    require(verification["scientific_verification"] == "PASS", "S3 result is not PASS")
    require(verification["points"] == 740, "S3 result does not report 740 points")
    actual_results = [
        {
            "threshold": float(row["threshold"]),
            "covered_points": int(row["covered_points"]),
            "gt_points": int(row["gt_points"]),
        }
        for row in verification["sweep_results"]
    ]
    expected_results = load_json(args.contract / "JR04_SCIENTIFIC_CONTRACT.json")[
        "expected_s3"
    ]
    require(actual_results == expected_results, f"Unexpected S3 results: {actual_results}")
    require(
        verification["all_12_columns_match_all_740_saved_point_rows_tau050"],
        "The 740 saved tau=0.50 point rows no longer match",
    )
    require(verification["input_files_unchanged"], "Preserved runner reports input mutation")
    require(not verification["new_training"], "Runner reports new training")
    require(not verification["new_vision_inference"], "Runner reports new vision inference")

    point_after = tree_pins(args.point_inputs.resolve())
    candidate_after = tree_pins(args.candidate_inputs.resolve())
    require(point_before == point_after, "Point input bytes changed during execution")
    require(candidate_before == candidate_after, "Candidate input bytes changed during execution")

    # Keep a compact, path-free copy of the scientific result.  The raw helper
    # directory is retained locally as audit evidence but is not intended for a
    # distributable review ZIP because its invocation receipts contain local paths.
    json_dump(
        out / "JR04_SANITIZED_S3_ACCEPTANCE.json",
        {
            "schema": "jr04-sanitized-s3-acceptance/v1",
            "status": "PASS",
            "preflight": preflight_result,
            "s3_results": actual_results,
            "tau050_all_12_columns_match_all_740_saved_rows": True,
            "candidate_population_checked_not_replotted": 10097,
            "historical_552_origin": "UNKNOWN",
            "current_tau080_saved_proposal_replay": 551,
            "models_loaded_or_unpickled": 0,
            "training_fitting_rescoring_image_inference_annotations_bootstrap": 0,
            "input_bytes_unchanged": True,
            "command_receipt": "S3_COMMAND_RECEIPT.json",
            "raw_local_helper_output": "EXCLUDE_FROM_DISTRIBUTABLE_USE_PORTABLE_RECEIPTS",
        },
    )
    print("JR04_SANITIZED_S3_ACCEPTANCE_PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"JR04_SANITIZED_ADAPTER_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
