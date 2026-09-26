"""Project-owned, fixed-feature r92 review snapshots and fresh XGBoost fits.

This is a new operational project lineage. It never changes the native r92
archive and makes no historical r92 model-selection claim.
"""

from __future__ import annotations

import csv
import base64
import hashlib
import json
import math
import shutil
from pathlib import Path

from .r92_sample import ARCHIVE_SHA256, _payloads, load_model, read_feature_csv

SCHEMA = "compag-r92-project-model/v1"
SNAPSHOT_SCHEMA = "compag-r92-project-training-snapshot/v2"
SNAPSHOT_SCHEMA_V1 = "compag-r92-project-training-snapshot/v1"
ACTIONS = {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4,
           "sus_flip": 0.4, "skip": 0.0, "delete": 0.0,
           "bulk_accept": 1.0}


def select_review(inference: Path, output: Path, *, count: int = 50) -> dict:
    """Select a bound uncertainty batch using the active decision policy."""
    inference, output = Path(inference), Path(output)
    if output.exists() or type(count) is not int or count < 1:
        raise ValueError("review selection output must be new and count positive")
    full = _json(inference / "FULL_INFERENCE_RECEIPT.json")
    scored = inference / "scores" / "detections.csv"
    if full.get("schema") != "compag-r92-infer-full/v1" or full.get("status") != "PASS" \
            or full.get("expected_tile_count") != full.get("processed_tile_count") \
            or _sha(scored) != full.get("score_receipt", {}).get("detections_sha256"):
        raise ValueError("review selection requires complete bound full-card scores")
    with scored.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    from .r92_yolo import PAPER_HYBRID_POLICY
    paper_hybrid = full["score_receipt"].get("fusion_policy") == PAPER_HYBRID_POLICY
    if len(rows) != full["candidate_count"] or (count > len(rows) and not paper_hybrid):
        raise ValueError("review selection count exceeds scored candidates")
    keys = [(r["image"],r["id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("review selection source has duplicate candidate keys")
    if paper_hybrid:
        # Section 2.10: rank by |p_ui - tau_eff| within al_margin=0.20.
        ranked = sorted(rows, key=lambda r:(abs(float(r["p_ui"])-float(r["final_threshold"])),
                                            r["image"],int(r["id"])))
        chosen = [r for r in ranked if abs(float(r["p_ui"])-float(r["final_threshold"])) <= .20][:count]
    else:
        chosen = sorted(rows, key=lambda r:(abs(float(r["xgb_p"])-float(r["threshold"])),
                                           r["image"],int(r["id"])))[:count]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image","id"])
        writer.writeheader()
        writer.writerows({"image":r["image"],"id":r["id"]} for r in chosen)
    return {"schema":"compag-r92-review-selection/v1","status":"PASS",
            "policy":("paper_smallest_absolute_p_ui_minus_tau_eff_within_al_margin"
                      if paper_hybrid else "smallest_absolute_raw_xgb_distance_to_threshold"),
            "al_margin": .20 if paper_hybrid else None,
            "selection_sha256":_sha(output),"score_sha256":_sha(scored),
            "selected_count":len(chosen),"context_count":len(rows)-len(chosen)}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _candidate_id(original_sha: str, tile_sha: str, mask_sha: str, tile: str, det_id: str) -> str:
    payload = json.dumps([original_sha, tile_sha, mask_sha, tile, det_id], separators=(",", ":"))
    return hashlib.sha256(("compag-project-candidate/v1\0" + payload).encode()).hexdigest()


def _review_events(path: Path) -> dict[tuple[str, str], dict]:
    """The last physical event wins; timestamps remain authentic provenance."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "id", "action", "human_label", "timestamp"} <= set(reader.fieldnames or ()):
            raise ValueError("review event log lacks required fields")
        events: dict[tuple[str, str], dict] = {}
        for sequence, row in enumerate(reader):
            action = row["action"].strip()
            if action not in ACTIONS:
                continue  # UI navigation/undo does not create a new effective label.
            key = (row["image"], str(row["id"]))
            if not all(key):
                raise ValueError("review event key is empty")
            stamp = int(row["timestamp"])
            if action in {"skip", "delete"}:
                if row["human_label"] not in {"", "nan", "NaN"}:
                    raise ValueError("excluded candidate must not carry a class label")
                label = None
            else:
                if row["human_label"] not in {"0", "1", "0.0", "1.0"}:
                    raise ValueError("a reviewed candidate requires a binary human label")
                label = int(float(row["human_label"]))
            claimed = float(row.get("review_weight") or ACTIONS[action])
            if not math.isclose(claimed, ACTIONS[action], rel_tol=0, abs_tol=1e-9):
                raise ValueError("review event weight differs from its action")
            normalized = {"image": key[0], "id": key[1], "action": action,
                          "label": label, "review_factor": ACTIONS[action],
                          "timestamp": stamp, "sequence": sequence}
            events[key] = normalized
    return events


def review_plan(state: Path, *, scope: str = "reviewed", selection: Path | None = None) -> dict:
    """Read-only counts and identities for an explicit proposed commit scope."""
    from .r92_review import _read_state
    state = Path(state)
    record = _read_state(state)
    with Path(record["scored_csv"]).open(newline="", encoding="utf-8") as stream:
        available = {(r["image"], r["id"]) for r in csv.DictReader(stream)}
    events = _review_events(state / "review_labels.csv")
    if not set(events) <= available:
        raise ValueError("review events reference candidates outside the bound scored pool")
    if scope not in {"reviewed", "all", "locked"}:
        raise ValueError("scope must be reviewed, all, or locked")
    if record["schema"].endswith("/v1") and scope != "locked":
        raise ValueError("expand the restricted v1 state before full-pool finalization")
    if scope == "locked":
        if selection is None:
            raise ValueError("locked scope needs explicit --selection")
        selected = _selected_keys(selection, available)
        if record.get("selection") and Path(selection).resolve() != Path(record["selection"]).resolve():
            raise ValueError("locked selection differs from state selection")
    elif scope == "all":
        selected = available
    else:
        if selection is not None:
            raise ValueError("--selection is only valid with --scope locked")
        selected = set(events)
    missing = selected - set(events)
    if missing:
        raise ValueError(f"{scope} scope lacks {len(missing)} terminal decisions")
    shortlist = _selected_keys(Path(record["selection"]), available) if record.get("selection") else set()
    labeled = {key for key in selected if events[key]["action"] not in {"skip", "delete"}}
    skipped = {key for key in selected if events[key]["action"] == "skip"}
    deleted = {key for key in selected if events[key]["action"] == "delete"}
    bulk_accepted = {key for key in labeled if events[key]["action"] == "bulk_accept"}
    return {"schema":"compag-r92-review-plan/v1", "scope":scope,
            "candidate_count":len(available), "terminal_count":len(selected),
            "eligible_labeled_count":len(labeled), "skipped_count":len(skipped),
            "deleted_count":len(deleted), "bulk_accepted_count":len(bulk_accepted),
            "unreviewed_count":len(available-set(events)),
            "excluded_from_training_count":len(available-labeled),
            "reviewed_in_shortlist":len(selected & shortlist),
            "reviewed_outside_shortlist":len(selected - shortlist),
            "shortlist_count":len(shortlist),
            "shortlist_bound":bool(record.get("selection")),
            "included_keys":[list(k) for k in sorted(labeled)],
            "excluded_keys":[list(k) for k in sorted(available-labeled)]}


def _selected_keys(path: Path | None, available: set[tuple[str, str]]) -> set[tuple[str, str]]:
    if path is None:
        return available
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"image", "id"}:
            raise ValueError("selection CSV must contain exactly image,id")
        keys = [(row["image"], row["id"]) for row in reader]
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= available:
        raise ValueError("selection contains absent or duplicate candidates")
    return set(keys)


def finalize_review(state: Path, inference: Path, model: Path, output: Path, *,
                    selection: Path | None = None, parent_snapshot: Path | None = None,
                    review_origin: str, confirm_review_complete: bool,
                    scope: str = "reviewed", correction_revision: str | None = None) -> dict:
    """Seal eligible explicit decisions from a declared scope."""
    if not confirm_review_complete or review_origin not in {"human", "test_fixture"}:
        raise ValueError("explicit completed-review confirmation and origin are required")
    from .r92_review import _read_state, _full_inference_binding
    state, inference, model, output = map(Path, (state, inference, model, output))
    if output.exists():
        raise FileExistsError("snapshot output already exists")
    _payloads(model)
    record = _read_state(state)
    plan = review_plan(state, scope=scope, selection=selection)
    scored = Path(record["scored_csv"])
    if scored.resolve() != (inference / "scores" / "detections.csv").resolve():
        raise ValueError("review state does not belong to this inference")
    project = Path(record["tiles"]).parent
    binding = _full_inference_binding(scored, project / "tiles")
    if binding is None or record.get("full_inference_receipt") != binding:
        raise ValueError("only a complete, bound full-card inference may be finalized")
    full = _json(inference / "FULL_INFERENCE_RECEIPT.json")
    if full.get("status") != "PASS" or full.get("expected_tile_count") != full.get("processed_tile_count"):
        raise ValueError("full-card coverage is incomplete")
    features_path = inference / "features.csv"
    if _sha(features_path) != full.get("feature_csv_sha256"):
        raise ValueError("inference feature table changed")
    proposals_path = inference / "proposals.csv"
    if _sha(proposals_path) != full.get("proposal_csv_sha256"):
        raise ValueError("full proposal/mask table changed")
    order = load_model(model).order
    with features_path.open(newline="", encoding="utf-8") as stream:
        feature_reader = csv.DictReader(stream)
        if not {"img_folder", "image", "id", "mask_sha256", *order} <= set(feature_reader.fieldnames or ()):
            raise ValueError("full inference lacks mask-bound 93-feature rows")
        feature_rows = {(r["image"], r["id"]): r for r in feature_reader}
    with scored.open(newline="", encoding="utf-8") as stream:
        scored_rows = {(r["image"], r["id"]): r for r in csv.DictReader(stream)}
    if len(feature_rows) != full["candidate_count"] or set(feature_rows) != set(scored_rows):
        raise ValueError("feature/score candidate closure differs")
    with proposals_path.open(newline="", encoding="utf-8") as stream:
        proposals = {(r["tile_name"], r["proposal_index"]): r for r in csv.DictReader(stream)}
    if set(proposals) != set(feature_rows) or any(
            proposals[key]["mask_sha256"] != row["mask_sha256"]
            for key, row in feature_rows.items()):
        raise ValueError("full mask/proposal candidate closure differs")
    events = _review_events(state / "review_labels.csv")
    selected = (set(events) if scope == "reviewed" else set(scored_rows) if scope == "all"
                else _selected_keys(selection, set(scored_rows)))
    manifest = _json(project / "PROJECT_MANIFEST.json")
    originals = {r["original"]: r for r in _json(project / "PROJECT_RECEIPT.json")["images"]}
    tiles = {t["name"]: (r, t) for r in originals.values() for t in r["tiles"]}
    rows: dict[str, dict] = {}
    prior_sha = None
    if parent_snapshot is not None:
        parent_snapshot = Path(parent_snapshot)
        prior = verify_snapshot(parent_snapshot)
        if prior["review_origin"] != review_origin:
            raise ValueError("human and software-test review data cannot be silently mixed")
        prior_sha = _sha(parent_snapshot)
        rows = {r["candidate_id"]: r for r in prior["rows"]}
    reviewed_count = skipped_count = deleted_count = repeated_count = 0
    for key in sorted(selected):
        feature, score, event = feature_rows[key], scored_rows[key], events[key]
        source, tile = tiles[feature["image"]]
        if feature["img_folder"] != source["original"] or not source["original_sha256"] == manifest["sha256_by_relative_path"]["originals/" + source["original"]]:
            raise ValueError("feature row original identity differs")
        mask_sha = feature["mask_sha256"]
        if len(mask_sha) != 64 or any(c not in "0123456789abcdef" for c in mask_sha):
            raise ValueError("mask identity is invalid")
        candidate_id = _candidate_id(source["original_sha256"], tile["sha256"], mask_sha, *key)
        if event["action"] in {"skip", "delete"}:
            if candidate_id in rows:
                if not correction_revision:
                    raise ValueError("excluding a previously committed candidate needs an explicit correction revision")
                del rows[candidate_id]
            if event["action"] == "delete":
                deleted_count += 1
            else:
                skipped_count += 1
            continue
        machine_decision = int(score.get("final_pred") or score["xgb_pred"])
        expected_label = machine_decision if event["action"] in {"accept", "sus_accept", "bulk_accept"} else 1-machine_decision
        if event["label"] != expected_label:
            raise ValueError("human label disagrees with recorded accept/flip action")
        reviewed_count += 1
        values = [float(feature[name]) if feature[name] not in ("", "nan", "NaN") else float("nan") for name in order]
        if any(math.isinf(v) for v in values):
            raise ValueError("infinite feature value")
        row = {"candidate_id": candidate_id, "group_id": source["original"],
               "original_sha256": source["original_sha256"], "tile": key[0], "tile_sha256": tile["sha256"],
               "mask_sha256": mask_sha, "id": key[1], "label": event["label"],
               "feature_state": "FROZEN_NATIVE_R92", "split_role": "train_only",
               "action": event["action"], "review_factor": event["review_factor"],
               "prior_xgb_p": float(score["xgb_p"]), "prior_threshold": float(score["threshold"]),
               "bbox": [float(feature[n]) for n in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")],
               "geometry_sha256": hashlib.sha256(feature["poly"].encode()).hexdigest(),
               "features": [None if math.isnan(value) else value for value in values],
               "review_event": event}
        if candidate_id in rows:
            earlier = rows[candidate_id]
            if any(earlier[field] != row[field] for field in
                   ("label", "features", "geometry_sha256", "mask_sha256", "tile_sha256")):
                if not correction_revision:
                    raise ValueError("a prior candidate changed; create an explicit correction revision")
                rows[candidate_id] = row
                continue
            repeated_count += 1
            continue
        rows[candidate_id] = row
    if not rows:
        raise ValueError("no eligible reviewed labels for a fit")
    snapshot = {"schema": SNAPSHOT_SCHEMA, "status": "PASS", "review_origin": review_origin,
                "feature_order": list(order), "feature_state": "FROZEN_NATIVE_R92",
                "native_model_sha256": ARCHIVE_SHA256, "parent_snapshot_sha256": prior_sha,
                "project_receipt_sha256": _sha(project / "PROJECT_RECEIPT.json"),
                "inference_receipt_sha256": _sha(inference / "FULL_INFERENCE_RECEIPT.json"),
                "score_sha256": _sha(scored), "feature_csv_sha256": _sha(features_path),
                "proposal_csv_sha256": _sha(proposals_path),
                "review_events_sha256": _sha(state / "review_labels.csv"),
                "selection_sha256": _sha(selection) if selection else None,
                "review_scope":scope, "review_plan":plan,
                "correction_revision":correction_revision,
                "selected_count": len(selected), "context_count": len(scored_rows) - len(selected),
                "reviewed_count": reviewed_count, "skipped_count": skipped_count,
                "deleted_count": deleted_count,
                "repeated_candidate_count": repeated_count,
                "cumulative_eligible_count": len(rows), "rows": [rows[k] for k in sorted(rows)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, allow_nan=False, separators=(",", ":")) + "\n")
    return {**{k: snapshot[k] for k in ("schema", "status", "review_origin", "selected_count",
                                     "context_count", "reviewed_count", "skipped_count", "deleted_count", "repeated_candidate_count",
                                     "cumulative_eligible_count")},
            "review_scope":scope, "reviewed_in_shortlist":plan["reviewed_in_shortlist"],
            "reviewed_outside_shortlist":plan["reviewed_outside_shortlist"]}


def verify_snapshot(path: Path) -> dict:
    snap = _json(path)
    if snap.get("schema") not in {SNAPSHOT_SCHEMA, SNAPSHOT_SCHEMA_V1} or snap.get("status") != "PASS" or snap.get("native_model_sha256") != ARCHIVE_SHA256:
        raise ValueError("project training snapshot identity is invalid")
    order = snap.get("feature_order")
    rows = snap.get("rows")
    if not isinstance(order, list) or len(order) != 93 or not isinstance(rows, list) or len(rows) != snap.get("cumulative_eligible_count"):
        raise ValueError("project training snapshot schema differs")
    ids = [r["candidate_id"] for r in rows]
    if len(ids) != len(set(ids)) or any(len(r["features"]) != 93 or r["label"] not in (0, 1) for r in rows):
        raise ValueError("project training snapshot candidate closure differs")
    return snap


def _rle(mask) -> dict:
    """Portable uncompressed COCO RLE in column-major pixel order."""
    import numpy as np
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("original-coordinate mask is empty")
    flattened = binary.ravel(order="F")
    changes = np.flatnonzero(flattened[1:] != flattened[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flattened.size]))).astype(int).tolist()
    if flattened[0]:
        counts.insert(0, 0)
    return {"size": list(binary.shape), "counts": counts}


def export_original_coco(snapshot: Path, inference: Path, output: Path) -> dict:
    """Map reviewed proposal masks to source pixels without bbox substitution."""
    import cv2
    import numpy as np
    from .canonical.proposals import canonical_mask_sha256

    snapshot, inference, output = map(Path, (snapshot, inference, output))
    if output.exists():
        raise FileExistsError("original COCO output already exists")
    snap, full = verify_snapshot(snapshot), _json(inference / "FULL_INFERENCE_RECEIPT.json")
    if _sha(inference / "FULL_INFERENCE_RECEIPT.json") != snap["inference_receipt_sha256"]:
        raise ValueError("snapshot and full inference differ")
    proposals_path = inference / "proposals.csv"
    if _sha(proposals_path) != snap["proposal_csv_sha256"]:
        raise ValueError("proposal mask table changed")
    # The full receipt binds the prepared project and every source tile.
    project_path = Path(full["project_path"]).resolve(strict=True)
    if _sha(project_path / "PROJECT_RECEIPT.json") != snap["project_receipt_sha256"]:
        raise ValueError("prepared project changed")
    originals = _json(project_path / "original_coco.json")
    tiled = _json(project_path / "tiled_coco.json")
    orig_by_id = {im["id"]: im for im in originals["images"]}
    tile_by_name = {im["file_name"]: im for im in tiled["images"]}
    with proposals_path.open(newline="", encoding="utf-8") as stream:
        proposals = {(r["tile_name"], r["proposal_index"]): r for r in csv.DictReader(stream)}
    annotations = []
    masks_by_image = {}
    for row in snap["rows"]:
        key = (row["tile"], row["id"])
        proposal = proposals.get(key)
        if proposal is None or proposal["mask_sha256"] != row["mask_sha256"]:
            continue  # Cumulative earlier rows are not part of this inference.
        tile = tile_by_name.get(row["tile"])
        if tile is None:
            raise ValueError("reviewed tile missing from project")
        meta = tile["meta"]
        original = orig_by_id[meta["orig_image_id"]]
        if original["file_name"] != row["group_id"]:
            raise ValueError("reviewed original identity changed")
        encoded = base64.b64decode(proposal["mask_packbits_base64"], validate=True)
        bits = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8), bitorder="little")
        if bits.size < 512*512:
            raise ValueError("proposal mask length is invalid")
        tile_mask = bits[:512*512].reshape(512, 512)
        if canonical_mask_sha256(tile_mask) != row["mask_sha256"]:
            raise ValueError("reviewed full mask changed")
        local_to_source = np.asarray(meta["inverse_matrix"], dtype=np.float64) @ np.asarray(
            [[1, 0, meta["offset_x"]], [0, 1, meta["offset_y"]], [0, 0, 1]], dtype=np.float64)
        source_mask = cv2.warpPerspective(tile_mask, local_to_source,
            (original["width"], original["height"]), flags=cv2.INTER_NEAREST)
        yy, xx = np.nonzero(source_mask)
        if not len(xx):
            raise ValueError("reviewed mask vanished in original coordinates")
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)]
        for previous in masks_by_image.get(original["id"], []):
            if previous[0] != row["label"]:
                continue
            ox, oy, ow, oh = previous[1]
            ix0, iy0 = max(ox,bbox[0]), max(oy,bbox[1])
            ix1, iy1 = min(ox+ow,bbox[0]+bbox[2]), min(oy+oh,bbox[1]+bbox[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            other = previous[2][iy0-oy:iy1-oy, ix0-ox:ix1-ox]
            current = source_mask[iy0:iy1, ix0:ix1]
            intersection = int(np.logical_and(other, current).sum())
            union = int(previous[2].sum()) + int(source_mask.sum()) - intersection
            if intersection and intersection / (union + 1e-6) >= 0.8:
                raise ValueError("overlapping reviewed masks need explicit physical-object QA")
        bx, by, bw, bh = bbox
        masks_by_image.setdefault(original["id"], []).append((row["label"], bbox, source_mask[by:by+bh,bx:bx+bw].copy()))
        annotations.append({"id": len(annotations)+1, "image_id": original["id"],
            "category_id": 1 if row["label"] == 1 else 2, "bbox": bbox,
            "area": int(source_mask.sum()), "iscrowd": 0, "segmentation": _rle(source_mask),
            "meta": {"candidate_id": row["candidate_id"], "tile": row["tile"],
                     "proposal_index": row["id"], "mask_sha256": row["mask_sha256"],
                     "review_event": row["review_event"], "scope": "committed_reviewed_only"}})
    if not annotations:
        raise ValueError("this inference has no finalized reviewed annotations")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {**originals, "annotations": annotations}
    output.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
    return {"schema": "compag-r92-original-coco-export/v1", "status": "PASS",
            "annotation_count": len(annotations), "output_sha256": _sha(output),
            "scope": "committed_reviewed_only_not_exhaustive"}


def _weights(rows: list[dict]) -> tuple[list[float], list[dict]]:
    """Explicit small-data adaptation of retained Only_codes review factors."""
    factors = []
    output = []
    for r in rows:
        x, y, width, height = r["bbox"]
        if r["action"] not in ACTIONS or r["action"] in {"skip", "delete"} or not math.isclose(
                float(r["review_factor"]), ACTIONS[r["action"]], rel_tol=0, abs_tol=1e-9):
            raise ValueError("training review factor differs from effective human action")
        if not (0 <= float(r["prior_xgb_p"]) <= 1 and 0 < float(r["prior_threshold"]) < 1):
            raise ValueError("prior model probability or threshold is invalid")
        if width <= 0 or height <= 0 or any(not math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError("training bbox is invalid")
        border = 0.5 if x <= 4 or y <= 4 or x + width >= 508 or y + height >= 508 else 1.0
        tiny_negative = 0.6 if r["label"] == 0 and (width < 32 or height < 32 or width * height <= 1500) else 1.0
        correct = 0.7 if int(r["prior_xgb_p"] >= r["prior_threshold"]) == r["label"] else 1.0
        uncertainty = 0.7 if abs(r["prior_xgb_p"] - r["prior_threshold"]) <= 0.05 else 1.0
        parts = {"review": r["review_factor"], "prior_correct": correct,
                 "uncertainty": uncertainty, "border": border, "tiny_negative": tiny_negative}
        weight = math.prod(parts.values())
        if not 0 < weight <= 1:
            raise ValueError("training sample weight is invalid")
        output.append(weight)
        factors.append(parts)
    return output, factors


def train_small_data(snapshot: Path, native_model: Path, output: Path, *, n_estimators: int = 100) -> dict:
    """Real fresh CPU fit; no holdout, CV, performance, or paper-r93 claim."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    if xgb.__version__ != "2.1.1" or type(n_estimators) is not int or not 10 <= n_estimators <= 500:
        raise ValueError("small-data fit requires XGBoost 2.1.1 and 10..500 trees")
    snapshot, native_model, output = map(Path, (snapshot, native_model, output))
    if output.exists():
        raise FileExistsError("project model output already exists")
    snap = verify_snapshot(snapshot)
    payloads = _payloads(native_model)
    if snap["native_model_sha256"] != _sha(native_model):
        raise ValueError("snapshot and native transform asset differ")
    rows, order = snap["rows"], snap["feature_order"]
    if order != json.loads(payloads["feature_schema.json"])["feature_order"]:
        raise ValueError("snapshot predictor order differs from frozen native r92")
    matrix = np.asarray([r["features"] for r in rows], dtype=np.float32)
    labels = np.asarray([r["label"] for r in rows], dtype=np.int32)
    if set(labels.tolist()) != {0, 1} or np.isinf(matrix).any():
        raise ValueError("fresh fit needs real reviewed instances of both classes and no infinities")
    inherited = json.loads(payloads["imputer.json"])["statistics"]
    medians = []
    sources = {}
    for index, name in enumerate(order):
        observed = matrix[:, index][~np.isnan(matrix[:, index])]
        if observed.size:
            value = float(np.median(observed))
            sources[name] = "PROJECT_TRAIN_ONLY_MEDIAN"
        else:
            value = float(inherited[name])
            sources[name] = "FROZEN_NATIVE_R92_FALLBACK_ALL_MISSING"
        if not math.isfinite(value):
            raise ValueError("imputer statistic is nonfinite")
        medians.append(value)
    filled = np.where(np.isnan(matrix), np.asarray(medians, dtype=np.float32), matrix)
    weights, factors = _weights(rows)
    parameters = {"n_estimators": n_estimators, "max_depth": 4, "learning_rate": 0.05,
                  "subsample": 1.0, "colsample_bytree": 1.0, "objective": "binary:logistic",
                  "eval_metric": "logloss", "tree_method": "hist", "device": "cpu",
                  "random_state": 42, "n_jobs": 1, "missing": float("nan")}
    classifier = xgb.XGBClassifier(**parameters)
    classifier.fit(pd.DataFrame(filled, columns=order), labels,
                   sample_weight=np.asarray(weights, dtype=np.float32))
    output.mkdir(parents=True)
    try:
        classifier.get_booster().save_model(str(output / "classifier.ubj"))
        (output / "imputer.json").write_text(json.dumps({"feature_order": order,
            "statistics": dict(zip(order, medians)), "sources": sources}, indent=2) + "\n")
        for name in ("feature_schema.json", "feature_state.json", "pca32_components.npy",
                     "pca32_mean.npy", "prototype.npy", "class_map.json"):
            (output / name).write_bytes(payloads[name])
        recipe = {"schema": "compag-r92-project-fit-recipe/v1", "policy": "SMALL_DATA_OPERATIONAL_NO_HOLDOUT",
                  "fit": "FRESH_FROM_SCRATCH", "parameters": parameters,
                  "sample_weight_components": [dict(candidate_id=r["candidate_id"], **f)
                                               for r, f in zip(rows, factors, strict=True)],
                  "observed_groups": sorted({r["group_id"] for r in rows}),
                  "split_roles": {g: "train_only" for g in sorted({r["group_id"] for r in rows})},
                  "group_cv_performed": False, "heldout_test_performed": False,
                  "model_quality_claim": False, "review_origin": snap["review_origin"],
                  "snapshot_sha256": _sha(snapshot), "native_model_sha256": _sha(native_model),
                  "class_counts": {str(c): int((labels == c).sum()) for c in (0, 1)},
                  "fitted_boosting_rounds": classifier.get_booster().num_boosted_rounds(),
                  "xgboost_version": xgb.__version__}
        (output / "TRAINING_RECIPE.json").write_text(json.dumps(recipe, indent=2) + "\n")
        members = {p.name: _sha(p) for p in output.iterdir() if p.is_file()}
        manifest = {"schema": SCHEMA, "status": "PASS", "files": members,
                    "classifier_sha256": members["classifier.ubj"], "native_model_sha256": ARCHIVE_SHA256,
                    "feature_order": order, "threshold": 0.5, "training_policy": recipe["policy"],
                    "snapshot_sha256": recipe["snapshot_sha256"], "review_origin": snap["review_origin"]}
        (output / "PROJECT_MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        loaded = load_project_model(output)
        probe = loaded.predict([dict(zip(order, rows[0]["features"]))])
        if len(probe) != 1 or not 0 <= float(probe[0]) <= 1:
            raise RuntimeError("new project classifier cannot be reloaded")
        return {"schema": SCHEMA, "status": "PASS", "classifier_sha256": members["classifier.ubj"],
                "snapshot_sha256": recipe["snapshot_sha256"], "row_count": len(rows),
                "boosting_rounds": recipe["fitted_boosting_rounds"], "policy": recipe["policy"]}
    except Exception:
        shutil.rmtree(output)
        raise


def load_project_model(root: Path):
    from .r92_sample import R92Predictor
    import numpy as np
    import xgboost as xgb

    if xgb.__version__ != "2.1.1":
        raise ValueError("project model requires XGBoost 2.1.1")
    root = Path(root).resolve(strict=True)
    manifest = _json(root / "PROJECT_MODEL_MANIFEST.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "PASS" or manifest.get("native_model_sha256") != ARCHIVE_SHA256:
        raise ValueError("project model manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {p.name for p in root.iterdir() if p.name != "PROJECT_MODEL_MANIFEST.json"}:
        raise ValueError("project model inventory differs")
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"project model member changed: {name}")
    order = tuple(manifest["feature_order"])
    if len(order) != 93 or tuple(_json(root / "feature_schema.json")["feature_order"]) != order:
        raise ValueError("project feature schema differs")
    imp = _json(root / "imputer.json")
    if imp["feature_order"] != list(order):
        raise ValueError("project imputer order differs")
    medians = np.asarray([imp["statistics"][name] for name in order], dtype=np.float32)
    if not np.isfinite(medians).all():
        raise ValueError("project imputer is nonfinite")
    booster = xgb.Booster()
    booster.load_model(str(root / "classifier.ubj"))
    booster.set_param({"device": "cpu", "nthread": 1})
    if tuple(booster.feature_names or ()) != order or booster.num_features() != 93:
        raise ValueError("project classifier feature order differs")
    return R92Predictor(booster, order, medians, files["classifier.ubj"])


def score_project_csv(bundle: Path, input_csv: Path, output: Path) -> dict:
    import numpy as np
    predictor = load_project_model(bundle)
    rows, matrix = read_feature_csv(input_csv, predictor.order)
    scores = predictor.predict(matrix)
    output = Path(output)
    if output.exists():
        raise FileExistsError("new project score output required")
    output.mkdir(parents=True)
    try:
        scored = output / "detections.csv"
        fields = ["img_folder", "image", "id", "xgb_p", "xgb_pred", "threshold",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "poly"]
        with scored.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, value in zip(rows, scores, strict=True):
                writer.writerow({"img_folder": row["img_folder"], "image": row["image"],
                    "id": row["id"], "xgb_p": repr(float(value)),
                    "xgb_pred": int(value >= 0.5), "threshold": "0.5",
                    **{name: row.get(name, "") for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h", "poly")}})
        receipt = {"schema": "compag-project-score/v1", "status": "PASS",
                   "execution_variant": "cpu_tabular_xgboost_2.1.1",
                   "model_sha256": predictor.model_sha256,
                   "model_archive_sha256": ARCHIVE_SHA256,
                   "project_model_manifest_sha256": _sha(Path(bundle) / "PROJECT_MODEL_MANIFEST.json"),
                   "input_sha256": _sha(input_csv), "detections_sha256": _sha(scored),
                   "row_count": len(rows), "threshold": 0.5,
                   "scientific_accuracy_claim": False}
        (output / "SCORE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise
