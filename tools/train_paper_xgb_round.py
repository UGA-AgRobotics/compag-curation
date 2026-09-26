#!/usr/bin/env python3
"""Fit a new r92 project XGBoost model with the recovered paper settings.

This tool consumes a sealed project training snapshot. It is deliberately separate
from the guided small-data fit: insufficient independent card reviews fail closed.
It applies the recovered XGBoost training method to *new* project data; it does
not reproduce the historical r92 fit or certify biological performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


POLICY = "PAPER_SETTINGS_NEW_DATA_GROUPED"
SCHEMA = "compag-r92-project-paper-fit-recipe/v1"
REVIEW_WEIGHTS = {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4}
NATIVE_MEMBERS = (
    "feature_schema.json", "feature_state.json", "pca32_components.npy",
    "pca32_mean.npy", "prototype.npy", "class_map.json",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _plain(value: Any) -> Any:
    """Turn NumPy scalars and immutable training records into strict JSON."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _plain(item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("training result contains a nonfinite value")
    return value


def preflight(snapshot: Path, model: Path, output: Path) -> tuple[dict, dict]:
    """Check data and deterministic splits without creating output files."""
    import numpy as np
    from compag_curation.canonical.spec import CANONICAL_FEATURE_ORDER
    from compag_curation.canonical.training import (
        CanonicalTrainingConfig, _group_safe_inner_split,
        canonical_tile_balanced_split,
    )
    from compag_curation.r92_project_model import verify_snapshot
    from compag_curation.r92_sample import _payloads
    from compag_curation.training.xgb import build_cv_pairs

    snapshot, model, output = map(Path, (snapshot, model, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"paper-settings model output already exists: {output}")
    snap = verify_snapshot(snapshot)
    if snap.get("review_origin") != "human":
        raise ValueError("paper-settings fit requires human-origin review decisions")
    rows = snap["rows"]
    bulk = sum(row.get("action") == "bulk_accept" for row in rows)
    if bulk:
        raise ValueError(
            f"paper-settings fit rejects {bulk} bulk-accepted predictions: "
            "individually review and correct candidates before training"
        )
    if not rows or tuple(snap["feature_order"]) != CANONICAL_FEATURE_ORDER:
        raise ValueError("paper-settings fit requires exact native r92 predictor features")
    group_hashes: dict[str, str] = {}
    hash_groups: dict[str, str] = {}
    features: list[dict[str, float]] = []
    actions: list[str] = []
    weights: list[float] = []
    for index, row in enumerate(rows):
        action = row.get("action")
        if action not in REVIEW_WEIGHTS:
            raise ValueError(f"row {index} lacks an individual paper review action")
        if not math.isclose(float(row.get("review_factor", -1)), REVIEW_WEIGHTS[action],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"row {index} review weight differs from its action")
        event = row.get("review_event")
        if not isinstance(event, dict) or event.get("action") != action or event.get("label") != row["label"]:
            raise ValueError(f"row {index} differs from its sealed review event")
        group = row.get("group_id")
        original_hash = row.get("original_sha256")
        if not isinstance(group, str) or not group or not isinstance(original_hash, str) \
                or len(original_hash) != 64 or any(c not in "0123456789abcdef" for c in original_hash):
            raise ValueError(f"row {index} lacks a valid original-card identity")
        if group in group_hashes and group_hashes[group] != original_hash:
            raise ValueError("one original card name maps to multiple source hashes")
        if original_hash in hash_groups and hash_groups[original_hash] != group:
            raise ValueError("one original card hash maps to multiple group names")
        group_hashes[group], hash_groups[original_hash] = original_hash, group
        try:
            values = tuple(float(value) for value in row["features"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"row {index} has a missing/non-numeric feature") from exc
        if len(values) != len(CANONICAL_FEATURE_ORDER) or not all(math.isfinite(value) for value in values):
            raise ValueError(f"row {index} has a missing/nonfinite canonical feature")
        features.append(dict(zip(CANONICAL_FEATURE_ORDER, values, strict=True)))
        actions.append(action)
        weights.append(REVIEW_WEIGHTS[action])
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    groups = np.asarray([row["group_id"] for row in rows])
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("paper-settings fit requires individually reviewed examples of both classes")
    if len(group_hashes) < 6:
        raise ValueError(
            f"paper-settings fit needs at least six independent reviewed cards "
            f"(five train folds plus held-out card); found {len(group_hashes)}"
        )
    split = canonical_tile_balanced_split(groups)
    if len(split.train_groups) < 5:
        raise ValueError("paper-settings split has fewer than five independent training cards")
    train_indices = np.flatnonzero(np.isin(groups, tuple(split.train_groups)))
    test_indices = np.flatnonzero(np.isin(groups, tuple(split.test_groups)))
    if set(labels[train_indices].tolist()) != {0, 1} or set(labels[test_indices].tolist()) != {0, 1}:
        raise ValueError("paper-settings train and held-out card groups must each contain both classes")
    cv_pairs = build_cv_pairs(groups[train_indices], labels[train_indices], 5)
    if len(cv_pairs) != 5:
        raise ValueError("paper-settings fit requires five valid GroupKFold folds with both classes")
    config = CanonicalTrainingConfig(device="cuda", thread_count=1)
    for fold_index, (fold_train, _) in enumerate(cv_pairs, start=1):
        try:
            _group_safe_inner_split(labels[train_indices][fold_train],
                                    groups[train_indices][fold_train], config)
        except ValueError as exc:
            raise ValueError(f"paper-settings fold {fold_index} has no valid group-safe early-stop split") from exc
    try:
        _group_safe_inner_split(labels[train_indices], groups[train_indices], config)
    except ValueError as exc:
        raise ValueError("paper-settings final fit has no valid group-safe early-stop split") from exc
    payloads = _payloads(model)
    if _sha(model) != snap["native_model_sha256"] or \
            tuple(json.loads(payloads["feature_schema.json"])["feature_order"]) != CANONICAL_FEATURE_ORDER:
        raise ValueError("native r92 model differs from the reviewed training snapshot")
    report = {
        "schema": "compag-r92-paper-fit-preflight/v1", "status": "READY",
        "policy": POLICY, "snapshot_sha256": _sha(snapshot), "native_model_sha256": _sha(model),
        "reviewed_row_count": len(rows), "independent_card_count": len(group_hashes),
        "train_card_count": len(split.train_groups), "heldout_card_count": len(split.test_groups),
        "group_folds": len(cv_pairs), "class_counts": dict(Counter(map(str, labels.tolist()))),
        "execution_device_required": "cuda", "historical_r92_reproduction_claim": False,
    }
    context = {"snap": snap, "payloads": payloads, "features": features, "labels": labels,
               "groups": groups, "actions": actions, "weights": weights, "split": split,
               "config": config, "train_indices": train_indices, "test_indices": test_indices}
    return report, context


def fit(snapshot: Path, model: Path, output: Path) -> dict:
    """Run the sealed CUDA paper-settings search and publish a portable model."""
    from compag_curation.canonical.training import train_canonical_xgb
    from compag_curation.r92_project_model import SCHEMA as MODEL_SCHEMA, load_project_model
    from compag_curation.r92_sample import ARCHIVE_SHA256

    snapshot, model, output = map(Path, (snapshot, model, output))
    report, source = preflight(snapshot, model, output)
    result = train_canonical_xgb(
        source["features"], source["labels"], source["groups"],
        source["actions"], source["weights"], split=source["split"],
        config=source["config"],
    )
    if tuple(result.feature_order) != tuple(source["snap"]["feature_order"]) or \
            result.search.fold_count != 5 or result.search.sampled_configuration_count != 30:
        raise RuntimeError("paper-settings trainer returned a different feature/search contract")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"paper-settings model output appeared during fitting: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.paper-fit-", dir=output.parent))
    try:
        booster = result.classifier.get_booster()
        booster.feature_names = list(result.feature_order)
        booster.save_model(str(stage / "classifier.ubj"))
        order = list(result.feature_order)
        _write_json(stage / "imputer.json", {
            "feature_order": order,
            "statistics": dict(zip(order, result.imputer_statistics, strict=True)),
            "sources": {name: "PAPER_SETTINGS_TRAIN_ONLY_MEDIAN" for name in order},
        })
        for name in NATIVE_MEMBERS:
            (stage / name).write_bytes(source["payloads"][name])
        split = result.split
        _write_json(stage / "GROUP_SPLIT.json", {
            "schema": "compag-r92-paper-group-split/v1", "status": "PASS",
            "unit": "original_full_card", "random_state": 42, "test_fraction": 0.20,
            "train_groups": sorted(split.train_groups), "heldout_groups": sorted(split.test_groups),
            "target_heldout_rows": split.target_test_rows,
            "observed_heldout_rows": split.observed_test_rows,
            "snapshot_sha256": report["snapshot_sha256"],
        })
        _write_json(stage / "SEARCH_RESULT.json", {
            "schema": "compag-r92-paper-xgb-search/v1", "status": "PASS",
            "group_folds": result.search.fold_count,
            "sampled_configuration_count": result.search.sampled_configuration_count,
            "best_parameters": _plain(dict(result.search.best_parameters)),
            "fold_mean_ap_by_configuration": _plain(result.search.score_ledger),
            "best_group_cv_mean_ap": float(result.search.best_average_precision),
            "selected_without_heldout_test": True,
        })
        heldout = source["test_indices"].tolist()
        if len(heldout) != len(result.test_probabilities):
            raise RuntimeError("paper-settings held-out prediction alignment differs")
        _write_json(stage / "HELDOUT_RESULT.json", {
            "schema": "compag-r92-paper-heldout/v1", "status": "PASS",
            "scope": "observed_new_project_cards_only", "performance_claim_for_other_cards": False,
            "metrics": _plain(dict(result.test_metrics)),
            "fixed_threshold": result.fixed_threshold,
            "rows": [{"candidate_id": source["snap"]["rows"][index]["candidate_id"],
                      "group_id": source["groups"][index].item(),
                      "label": int(label), "probability": float(probability)}
                     for index, label, probability in zip(
                         heldout, result.test_labels, result.test_probabilities, strict=True)],
        })
        recipe = {
            "schema": SCHEMA, "status": "PASS", "policy": POLICY,
            "fit": "FRESH_FROM_SCRATCH", "review_origin": "human",
            "snapshot_sha256": report["snapshot_sha256"],
            "native_model_sha256": ARCHIVE_SHA256,
            "paper_reference": {"section": "2.8, 2.10, Table 2, Supplementary Table S1",
                                "pdf_pages": [6, 7, 11, 20, 21]},
            "method": {"unit": "original_full_card", "group_folds": 5,
                       "random_search_iterations": 30, "inner_validation_fraction": 0.30,
                       "early_stopping_rounds": 30, "safe_smote": "training_fold_only_when_feasible",
                       "maximum_refit_estimators": 2000, "random_state": 42,
                       "eval_metric": "aucpr", "scale_pos_weight": 1.0,
                       "decision_threshold": 0.50, "device": "cuda",
                       "used_trees": result.used_trees,
                       "review_weights": REVIEW_WEIGHTS},
            "data_scope": "new_project_review_snapshot_single_scale_features",
            "historical_r92_reproduction_claim": False,
            "independent_scientific_validation_claim": False,
        }
        _write_json(stage / "TRAINING_RECIPE.json", recipe)
        files = {p.name: _sha(p) for p in sorted(stage.iterdir()) if p.is_file()}
        _write_json(stage / "PROJECT_MODEL_MANIFEST.json", {
            "schema": MODEL_SCHEMA, "status": "PASS", "files": files,
            "classifier_sha256": files["classifier.ubj"],
            "native_model_sha256": ARCHIVE_SHA256, "feature_order": order,
            "threshold": 0.50, "training_policy": POLICY,
            "snapshot_sha256": report["snapshot_sha256"], "review_origin": "human",
        })
        loaded = load_project_model(stage)
        probe = loaded.predict([source["features"][0]])
        if len(probe) != 1 or not 0 <= float(probe[0]) <= 1:
            raise RuntimeError("new paper-settings model could not be reloaded")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"paper-settings model output appeared during packaging: {output}")
        stage.rename(output)
        return {"schema": "compag-r92-paper-fit-result/v1", "status": "PASS",
                "policy": POLICY, "model_dir": str(output.resolve()),
                "classifier_sha256": files["classifier.ubj"],
                "snapshot_sha256": report["snapshot_sha256"],
                "group_folds": 5, "sampled_configuration_count": 30,
                "used_trees": result.used_trees,
                "heldout_card_count": len(split.test_groups),
                "historical_r92_reproduction_claim": False}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="verified native r92 model ZIP")
    parser.add_argument("--output", type=Path, required=True,
                        help="new project XGBoost model directory")
    parser.add_argument("--preflight", action="store_true",
                        help="check reviewed labels and card splits without fitting or writing")
    args = parser.parse_args(argv)
    try:
        receipt = (preflight(args.snapshot, args.model, args.output)[0]
                   if args.preflight else fit(args.snapshot, args.model, args.output))
    except Exception as exc:
        print(f"Paper-settings XGBoost unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
