"""Bridge newly computed r92 table scores to the existing local reviewer/merge."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import secrets
import shutil
from pathlib import Path

from .only_codes.review import GuiParams, LegacyReviewSession, append_review_row, append_review_rows, key_str, load_labels_effective, parse_tile, compute_final_pred_and_ui, is_uncertain, load_seen
from .only_codes.review_web import run_review_web
from .only_codes.merge import merge_reviews
from .r92_yolo import LEGACY_FUSION_POLICY, PAPER_HYBRID_POLICY

TILE_SIZE = 512  # The frozen merge emits this size.
CATEGORIES = {1: "CJ", 2: "non-CJ"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preflight_rows(scored: Path, tiles: Path) -> list[dict[str, str]]:
    """Reject ambiguous keys, absent pixels and unusable geometry before state exists."""
    import cv2

    with scored.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"img_folder", "image", "id", "xgb_p", "poly", "bbox_x", "bbox_y", "bbox_w", "bbox_h"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("r92 score table lacks review fields")
        rows = list(reader)
    if not rows:
        raise ValueError("r92 review needs at least one candidate")
    keys: set[str] = set()
    checked_images: set[str] = set()
    for row in rows:
        name, det_id = row["image"], row["id"]
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", det_id or ""):
            raise ValueError("candidate id must be a canonical nonnegative integer")
        key = key_str(name, det_id)
        if key in keys:
            raise ValueError("duplicate downstream (image,id) key, including img_folder collision")
        keys.add(key)
        parsed = parse_tile(name)
        if not row["img_folder"] or parsed is None or parsed[3].lower() not in {"jpg", "jpeg", "png"} or Path(name).name != name:
            raise ValueError("candidate image must be a supported prepared tile basename")
        if name not in checked_images:
            tile = tiles / name
            if tile.is_symlink() or not tile.is_file() or tile.resolve().parent != tiles:
                raise ValueError(f"missing or unsafe referenced tile: {name}")
            pixels = cv2.imread(str(tile), cv2.IMREAD_UNCHANGED)
            if pixels is None or pixels.shape[:2] != (TILE_SIZE, TILE_SIZE):
                raise ValueError(f"prepared tile must be readable 512x512: {name}")
            checked_images.add(name)
        try:
            p = float(row["xgb_p"])
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate probability is invalid") from exc
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("candidate probability is invalid")
        raw = row.get("poly", "").strip()
        if raw:
            try:
                polygon = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("candidate polygon is malformed") from exc
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("candidate polygon is malformed")
        else:
            try:
                x, y, w, h = (float(row[n]) for n in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            except (TypeError, ValueError) as exc:
                raise ValueError("candidate lacks a valid polygon or disclosed bbox fallback") from exc
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError("candidate bbox fallback is invalid")
            polygon = [x, y, x+w, y, x+w, y+h, x, y+h]
        try:
            coords = [float(v) for v in polygon]
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate geometry is nonnumeric") from exc
        if not all(math.isfinite(v) and 0 <= v <= TILE_SIZE for v in coords):
            raise ValueError("candidate geometry is outside the 512x512 tile")
        xs, ys = coords[0::2], coords[1::2]
        twice_area = sum(xs[i]*ys[(i+1)%len(xs)] - xs[(i+1)%len(xs)]*ys[i] for i in range(len(xs)))
        if max(xs) == min(xs) or max(ys) == min(ys) or abs(twice_area) <= 1e-9:
            raise ValueError("candidate geometry has zero area")
    return rows


def _registry(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(isinstance(data.get(n), list) for n in ("images", "annotations", "categories")):
        raise ValueError("COCO registry is malformed")
    if len(data["categories"]) != 2 or {c.get("id"): c.get("name") for c in data["categories"]} != CATEGORIES:
        raise ValueError("COCO categories must map 1=CJ and 2=non-CJ")
    ids = [im.get("id") for im in data["images"]]
    names = [im.get("file_name") for im in data["images"]]
    if any(type(v) is not int or v <= 0 for v in ids) or len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("COCO image IDs or names are invalid or duplicated")
    return data


def _check_registries(rows: list[dict[str, str]], orig: dict, tiled: dict, orig_coco: Path) -> None:
    import cv2

    originals = {Path(i["file_name"]).stem: i for i in orig["images"]}
    tilemap = {i["file_name"]: i for i in tiled["images"]}
    checked_originals: set[str] = set()
    if len(originals) != len(orig["images"]):
        raise ValueError("original image stems are ambiguous")
    for row in rows:
        name = row["image"]
        root, y, x, _ = parse_tile(name)
        oi, ti = originals.get(root), tilemap.get(name)
        if oi is None or ti is None:
            raise ValueError("COCO registries must contain the original and tile")
        ow, oh = oi.get("width"), oi.get("height")
        if not all(type(v) is int and v > 0 for v in (ow, oh)):
            raise ValueError("original image dimensions are invalid")
        original_file = orig_coco.parent / "originals" / oi["file_name"]
        if original_file.is_symlink() or not original_file.is_file() or original_file.resolve().parent != (orig_coco.parent / "originals").resolve():
            raise ValueError("registered original image is missing or unsafe")
        if oi["file_name"] not in checked_originals:
            original_pixels = cv2.imread(str(original_file), cv2.IMREAD_UNCHANGED)
            if original_pixels is None or original_pixels.shape[:2] != (oh, ow):
                raise ValueError("original COCO dimensions differ from actual pixels")
            checked_originals.add(oi["file_name"])
        if ti.get("width") != TILE_SIZE or ti.get("height") != TILE_SIZE:
            raise ValueError("COCO tile dimensions do not match actual pixels")
        meta = ti.get("meta") or {}
        if (meta.get("orig_image_id"), meta.get("orig_file_name"), meta.get("offset_x"), meta.get("offset_y")) != (oi["id"], oi["file_name"], x, y):
            raise ValueError("COCO tile original link or offsets are invalid")
        ww, wh = meta.get("warped_width"), meta.get("warped_height")
        matrix = meta.get("inverse_matrix")
        if (type(ww) is not int or type(wh) is not int or x + TILE_SIZE > ww or y + TILE_SIZE > wh
                or meta.get("warp_mode") not in {"identity_fallback", "yellow_card_quadrilateral"}
                or not isinstance(matrix, list) or len(matrix) != 3
                or any(not isinstance(line, list) or len(line) != 3 for line in matrix)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for line in matrix for v in line)):
            raise ValueError("COCO canonical warp dimensions or inverse transform are invalid")


def _read_state(state: Path) -> dict:
    payload = json.loads((state / "R92_REVIEW_STATE.json").read_text())
    if payload.get("schema") not in {"compag-r92-review-state/v1", "compag-r92-review-state/v2"}:
        raise ValueError("r92 review state schema changed")
    score = Path(payload["scored_csv"])
    if not score.is_file() or _sha(score) != payload["scored_sha256"]:
        raise ValueError("review source scores changed")
    receipt = json.loads((score.parent / "SCORE_RECEIPT.json").read_text())
    if (receipt.get("detections_sha256") != payload["scored_sha256"]
            or receipt.get("model_sha256") != payload.get("model_sha256")
            or receipt.get("threshold") != payload.get("threshold")
            or receipt.get("execution_variant") != "cpu_tabular_xgboost_2.1.1"):
        raise ValueError("r92 score receipt does not bind this review")
    _preflight_rows(score, Path(payload["tiles"]))
    if "selection" in payload:
        selected = Path(payload["selection"])
        if not selected.is_file() or _sha(selected) != payload.get("selection_sha256"):
            raise ValueError("review selection changed")
        if payload["schema"].endswith("/v1"):
            subset = state / "review_subset.csv"
            if not subset.is_file() or _sha(subset) != payload.get("review_subset_sha256"):
                raise ValueError("selected review table changed")
    if "full_inference_receipt" in payload:
        binding = _full_inference_binding(score, Path(payload["tiles"]))
        if binding is None or binding != payload["full_inference_receipt"]:
            raise ValueError("full inference receipt or project changed")
    return payload


def _full_inference_binding(scored: Path, tiles: Path) -> dict | None:
    """Bind a scored full round to every prepared tile, including empty tiles."""
    receipt_path = scored.parent.parent / "FULL_INFERENCE_RECEIPT.json"
    if not receipt_path.is_file():
        return None
    full = json.loads(receipt_path.read_text())
    project = tiles.resolve().parent
    project_receipt = project / "PROJECT_RECEIPT.json"
    orig_coco = project / "original_coco.json"
    tiled_coco = project / "tiled_coco.json"
    tiled = _registry(tiled_coco)
    names = {im["file_name"] for im in tiled["images"]}
    processed = full.get("tiles")
    if (full.get("schema") != "compag-r92-infer-full/v1" or full.get("status") != "PASS"
            or full.get("full_prepared_set") is not True
            or full.get("expected_tile_count") != len(names)
            or full.get("processed_tile_count") != len(names)
            or not isinstance(processed, list) or len(processed) != len(names)
            or {r.get("tile") for r in processed} != names
            or full.get("score_receipt", {}).get("detections_sha256") != _sha(scored)
            or full.get("score_receipt", {}).get("input_sha256") != full.get("feature_csv_sha256")
            or full.get("r92_model_archive_sha256") != full.get("score_receipt", {}).get("model_archive_sha256")
            or full.get("project_receipt_sha256") != _sha(project_receipt)
            or full.get("original_coco_sha256") != _sha(orig_coco)
            or full.get("tiled_coco_sha256") != _sha(tiled_coco)):
        raise ValueError("full inference is incomplete or differs from this project")
    return {"path": str(receipt_path.resolve()), "sha256": _sha(receipt_path),
            "project": str(project), "tile_count": len(names)}


def effective_events(path: Path) -> dict[str, dict]:
    """Current r92 decision by append order; skip supersedes an earlier label."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "id", "action", "human_label", "timestamp"} <= set(reader.fieldnames or ()):
            raise ValueError("review event log lacks required fields")
        out = {}
        for sequence, row in enumerate(reader):
            if row["action"] not in {"accept", "flip", "sus_accept", "sus_flip", "skip", "delete", "bulk_accept", "auto_accept"}:
                raise ValueError("unexpected r92 review action")
            key = key_str(row["image"], row["id"])
            if not row["image"] or not row["id"] or not row["timestamp"].isdigit():
                raise ValueError("review event identity or timestamp is invalid")
            if row["action"] == "auto_accept":
                continue  # Older script action is not a committed human or bulk decision.
            out[key] = {**row, "sequence": sequence}
        return out


class R92ReviewSession(LegacyReviewSession):
    """Project adapter: immutable model facts plus current human event state."""

    def bulk_plan(self) -> dict:
        """Preview every undecided candidate across the bound scored pool."""
        events = effective_events(self.review_csv)
        remaining = self.df[~self.df.apply(
            lambda row: key_str(str(row["image"]), str(row["id"])) in events, axis=1)]
        counts = {"CJ": 0, "non-CJ": 0}
        for _, row in remaining.iterrows():
            pred, _, _ = compute_final_pred_and_ui(row.to_dict(), self.params)
            counts["CJ" if pred else "non-CJ"] += 1
        return {"schema": "compag-r92-bulk-accept-plan/v1", "remaining_count": len(remaining),
                "model_cj_count": counts["CJ"], "model_noncj_count": counts["non-CJ"],
                "already_decided_count": len(events), "pool_count": len(self.df),
                "review_log_sha256": _sha(self.review_csv) if self.review_csv.is_file() else None,
                "scope": "all original cards in this bound scored pool"}

    def act(self, action: str, sel: str | None = None, base: str | None = None,
            expected_count: int | None = None, expected_log_sha256: str | None = None) -> str:
        if action == "delete":
            if not sel or self._row_for(sel) is None:
                raise ValueError("select a valid scored candidate to delete")
            tile, det_id = sel.split("||", 1)
            record = self._record(tile, det_id, float("nan"), "delete", self._row_for(sel))
            record["review_weight"] = 0.0
            append_review_row(self.review_csv, record)
            self.labels_eff = load_labels_effective(self.review_csv)
            return f"[DELETE] {sel} excluded from the active review pool and downstream training/export. Undo last event restores its previous decision."
        if action == "accept_remaining":
            plan = self.bulk_plan()
            count = plan["remaining_count"]
            if (type(expected_count) is not int or expected_count != count
                    or expected_log_sha256 != plan["review_log_sha256"]):
                raise ValueError("remaining count changed; preview and confirm again")
            if count == 0:
                return "[BULK] No undecided candidates remain."
            events = effective_events(self.review_csv)
            records = []
            for _, rr in self.df.iterrows():
                row = rr.to_dict()
                tile, det_id = str(row["image"]), str(row["id"])
                if key_str(tile, det_id) in events:
                    continue
                pred, _, _ = compute_final_pred_and_ui(row, self.params)
                records.append(self._record(tile, det_id, pred, "bulk_accept", row))
            if len(records) != count:
                raise ValueError("remaining candidate count changed")
            batch_id = secrets.token_hex(12)
            batch_dir = self.review_csv.parent / "bulk_accept_batches"
            batch_dir.mkdir(exist_ok=True)
            before = self.review_csv.read_bytes() if self.review_csv.exists() else None
            backup = batch_dir / f"{batch_id}.before.csv"
            if before is not None:
                backup.write_bytes(before)
            append_review_rows(self.review_csv, records)
            receipt = {"schema": "compag-r92-bulk-accept/v1", "batch_id": batch_id,
                       "count": count, "model_cj_count": plan["model_cj_count"],
                       "model_noncj_count": plan["model_noncj_count"],
                       "before_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
                       "after_sha256": _sha(self.review_csv), "backup": str(backup) if before is not None else None,
                       "action": "user_confirmed_model_predictions", "review_weight": 1.0}
            (batch_dir / f"{batch_id}.json").write_text(json.dumps(receipt, indent=2) + "\n")
            (batch_dir / "LATEST.json").write_text(json.dumps({"batch_id": batch_id}) + "\n")
            self.labels_eff = load_labels_effective(self.review_csv)
            return f"[BULK] Accepted {count} previously undecided model predictions across all cards (CJ={plan['model_cj_count']}, non-CJ={plan['model_noncj_count']}). Undo last batch is available until another decision changes the event log."
        if action == "undo_bulk":
            batch_dir = self.review_csv.parent / "bulk_accept_batches"
            latest = batch_dir / "LATEST.json"
            if not latest.is_file():
                raise ValueError("no bulk acceptance to undo")
            batch_id = json.loads(latest.read_text())["batch_id"]
            receipt = json.loads((batch_dir / f"{batch_id}.json").read_text())
            if (batch_dir / f"{batch_id}.undone.json").exists():
                raise ValueError("latest bulk acceptance was already undone")
            if not self.review_csv.is_file() or _sha(self.review_csv) != receipt["after_sha256"]:
                raise ValueError("review log changed after this batch; revise candidates individually")
            if receipt["before_sha256"] is None:
                self.review_csv.unlink()
            else:
                backup = batch_dir / f"{batch_id}.before.csv"
                if _sha(backup) != receipt["before_sha256"]:
                    raise ValueError("bulk backup changed")
                self.review_csv.write_bytes(backup.read_bytes())
            (batch_dir / f"{batch_id}.undone.json").write_text(json.dumps({
                "schema": "compag-r92-bulk-undo/v1", "batch_id": batch_id,
                "restored_sha256": receipt["before_sha256"]}, indent=2) + "\n")
            self.labels_eff = load_labels_effective(self.review_csv)
            return f"[UNDO BULK] Restored the review log before batch {batch_id} ({receipt['count']} decisions)."
        return super().act(action, sel=sel, base=base)

    def items(self, base: str) -> list[dict]:
        items = super().items(base)
        events = effective_events(self.review_csv)
        rows = {key_str(str(r["image"]), str(r["id"])): r.to_dict()
                for _, r in self.df[self.df["base"] == base].iterrows()}
        for it in items:
            row = rows[it["key"]]
            pred, p_ui, threshold = compute_final_pred_and_ui(row, self.params)
            event = events.get(it["key"])
            action = event["action"] if event else None
            human = int(float(event["human_label"])) if action in {"accept", "flip", "sus_accept", "sus_flip", "bulk_accept"} else None
            it.update(pred=pred, p_ui=p_ui, thr_eff=threshold,
                      uncertain=is_uncertain(p_ui, threshold, self.params),
                      raw_xgb_pred=int(float(row["xgb_p"]) >= float(self.params.thr_xgb)),
                      raw_xgb_hard=abs(float(row["xgb_p"])-float(self.params.thr_xgb)) <= self.params.al_margin,
                      score_source="hybrid_effective" if self.params.use_yolo else "raw_xgb",
                      boundary_distance=None if p_ui is None else abs(p_ui-threshold),
                      shortlisted=it["key"] in self.project_selection,
                      shortlist_rank=self.project_selection.get(it["key"]),
                      shortlist_reason=self.project_selection_reason if it["key"] in self.project_selection else None,
                      reviewed=action is not None, human=human, action=action,
                      review_status=("deleted" if action == "delete" else
                                     "skipped" if action == "skip" else
                                     "bulk_accepted" if action == "bulk_accept" else
                                     "human_uncertain" if action in {"sus_accept", "sus_flip"} else
                                     "confident" if action else "unreviewed"),
                      mismatch=human is not None and human != pred)
        return items


def _selection_keys(selection: Path, scored_keys: set[str], *, allow_empty: bool = False) -> dict[str, int]:
    with selection.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"image", "id"}:
            raise ValueError("review selection needs exactly image,id")
        keys = [key_str(r["image"], r["id"]) for r in reader]
    if (not keys and not allow_empty) or len(keys) != len(set(keys)) or not set(keys) <= scored_keys:
        raise ValueError("review selection is empty, duplicated, or outside the scored pool")
    return {key: rank for rank, key in enumerate(keys, 1)}


def make_session(scored: Path, tiles: Path, state: Path,
                 *, selection: Path | None = None) -> LegacyReviewSession:
    scored = Path(scored).resolve(strict=True)
    tiles = Path(tiles).resolve(strict=True)
    state = Path(state).resolve()
    if selection is None and (state / "R92_REVIEW_STATE.json").is_file():
        prior_selection = json.loads((state / "R92_REVIEW_STATE.json").read_text()).get("selection")
        if prior_selection:
            selection = Path(prior_selection)
    if not tiles.is_dir() or state == tiles or tiles in state.parents:
        raise ValueError("review state must be outside the tiles directory")
    receipt = json.loads((scored.parent / "SCORE_RECEIPT.json").read_text())
    if receipt.get("detections_sha256") != _sha(scored) or receipt.get("execution_variant") != "cpu_tabular_xgboost_2.1.1":
        raise ValueError("scores are not a verified r92 CPU output")
    threshold = receipt.get("threshold")
    if threshold not in (0.5, 0.5843676924705505):
        raise ValueError("unknown r92 review threshold")
    all_rows = _preflight_rows(scored, tiles)
    for row in all_rows:
        if float(row.get("threshold", "nan")) != float(threshold) or int(row.get("xgb_pred", "-1")) != int(float(row["xgb_p"]) >= float(threshold)):
            raise ValueError("recorded candidate threshold or raw XGBoost prediction differs from the score receipt")
    scored_keys = {key_str(r["image"], r["id"]) for r in all_rows}
    legacy_existing = (state / "R92_REVIEW_STATE.json").is_file() and json.loads((state / "R92_REVIEW_STATE.json").read_text()).get("schema") == "compag-r92-review-state/v1"
    record = {"schema":"compag-r92-review-state/v1" if legacy_existing else "compag-r92-review-state/v2",
              "scored_csv":str(scored),"scored_sha256":_sha(scored),"tiles":str(tiles),"threshold":threshold,"model_sha256":receipt["model_sha256"]}
    subset_rows = None
    selected_keys = {}
    if selection is not None:
        selection = Path(selection).resolve(strict=True)
        selected_keys = _selection_keys(selection, scored_keys,
                                        allow_empty=receipt.get("fusion_policy") == PAPER_HYBRID_POLICY)
        if legacy_existing:
            with scored.open(newline="", encoding="utf-8") as stream:
                fields = list(csv.DictReader(stream).fieldnames or ())
            by_key = {key_str(r["image"], r["id"]):r for r in all_rows}
            subset_rows = [by_key[key] for key in selected_keys]
        record.update({"selection": str(selection), "selection_sha256": _sha(selection)})
    full_binding = _full_inference_binding(scored, tiles)
    if full_binding is not None:
        record["full_inference_receipt"] = full_binding
    path = state / "R92_REVIEW_STATE.json"
    if path.exists():
        previous = _read_state(state)
        comparable = {k:v for k,v in previous.items() if k != "review_subset_sha256"}
        if comparable != record:
            raise ValueError("review state belongs to another score table or tiles path")
    else:
        if state.exists():
            raise FileExistsError("new review state destination already exists")
        state.mkdir(parents=True, exist_ok=False)
        if subset_rows is not None:
            subset_path = state / "review_subset.csv"
            with subset_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(subset_rows)
            record["review_subset_sha256"] = _sha(subset_path)
        path.write_text(json.dumps(record,indent=2)+"\n")
    fusion = receipt.get("yolo_mode") in {"fusion", "both"}
    fusion_policy = receipt.get("fusion_policy") if fusion else None
    if fusion and fusion_policy not in {None, LEGACY_FUSION_POLICY, PAPER_HYBRID_POLICY}:
        raise ValueError("unknown bound YOLO/XGBoost fusion policy")
    paper_hybrid = fusion_policy == PAPER_HYBRID_POLICY
    if paper_hybrid and float(threshold) != .50:
        raise ValueError("paper hybrid requires XGBoost threshold tau_xgb=0.50")
    params = (GuiParams(use_xgb=True, use_yolo=True, det_policy="hybrid",
                        det_missing="reject" if paper_hybrid else "ignore",
                        thr_xgb=float(threshold), thr_yolo=.20, thr_iou=.60, det_thr=.50,
                        hybrid_yolo_bias=.80) if fusion else
              GuiParams(use_xgb=True, use_yolo=False, det_policy="xgb", thr_xgb=float(threshold)))
    if fusion:
        for row in all_rows:
            pred, p_ui, thr_eff = compute_final_pred_and_ui(row, params)
            if int(row.get("final_pred", "-1")) != pred or float(row.get("final_threshold", "nan")) != thr_eff:
                raise ValueError("recorded effective YOLO/XGBoost decision differs from the bound score policy")
            if paper_hybrid:
                valid_yolo = (float(row.get("yolo_conf", "nan")) >= params.thr_yolo
                              and float(row.get("yolo_iou", "nan")) >= params.thr_iou)
                if (row.get("yolo_valid") != ("1" if valid_yolo else "0")
                        or not math.isclose(float(row.get("p_ui", "nan")), float(p_ui), abs_tol=1e-5)):
                    raise ValueError("recorded paper hybrid review score or YOLO support differs")
                if valid_yolo:
                    if (not math.isclose(float(row.get("p_fused", "nan")), float(p_ui), abs_tol=1e-5)
                            or row.get("fused_p") != row.get("p_fused")):
                        raise ValueError("recorded paper hybrid fused score differs")
                elif row.get("p_fused") != "" or row.get("fused_p") != "":
                    raise ValueError("a candidate without valid YOLO support has no fused score")
            elif not math.isclose(float(row.get("p_fused", "nan")), float(p_ui), abs_tol=1e-5):
                raise ValueError("recorded legacy YOLO/XGBoost fused score differs")
    displayed = state / "review_subset.csv" if legacy_existing and subset_rows is not None else scored
    cls = LegacyReviewSession if legacy_existing else R92ReviewSession
    session = cls(detections_csv=displayed,tiles_dir=tiles,review_csv=state/"review_labels.csv",seen_json=state/"seen.json",cache_dir=state/"cache",params=params)
    if not legacy_existing:
        session.project_selection = selected_keys
        if paper_hybrid:
            ranked = sorted(all_rows, key=lambda r: (
                abs(float(r["p_ui"])-float(r["final_threshold"])), r["image"], int(r["id"])))
            expected = [key_str(r["image"], r["id"]) for r in ranked
                        if abs(float(r["p_ui"])-float(r["final_threshold"])) <= .20]
            session.project_selection_reason = (
                "smallest |p_ui - tau_eff| within al_margin=0.20"
                if list(selected_keys) == expected[:len(selected_keys)] else "bound recommendation CSV")
        else:
            raw_order = [key_str(r["image"], r["id"]) for r in sorted(
                all_rows, key=lambda r: (abs(float(r["xgb_p"])-float(r["threshold"])),
                                         r["image"], int(r["id"])))]
            session.project_selection_reason = ("closest raw-XGB boundary distance"
                if list(selected_keys) == raw_order[:len(selected_keys)] else "bound recommendation CSV")
        session.project_pool_count = len(all_rows)
        session.project_ui = True
    return session


def expand_session(old_state: Path, new_state: Path) -> dict:
    """Copy authentic restricted events into a new full-pool state revision."""
    old_state = Path(old_state).resolve(strict=True)
    new_state = Path(new_state).resolve()
    old = _read_state(old_state)
    if old["schema"] != "compag-r92-review-state/v1" or "selection" not in old or "full_inference_receipt" not in old:
        raise ValueError("expansion requires a bound restricted v1 full-card session")
    if new_state.exists() or old_state == new_state or old_state in new_state.parents:
        raise ValueError("new state revision must be a new separate directory")
    keys = {key_str(r["image"], r["id"]) for r in _preflight_rows(Path(old["scored_csv"]), Path(old["tiles"]))}
    selected = set(_selection_keys(Path(old["selection"]), keys))
    events = effective_events(old_state / "review_labels.csv")
    seen_path = old_state / "seen.json"
    if seen_path.is_file():
        seen_raw = json.loads(seen_path.read_text())
        if not isinstance(seen_raw, list) or not all(isinstance(k, str) for k in seen_raw) or len(set(seen_raw)) != len(seen_raw):
            raise ValueError("old seen navigation is malformed")
    if not set(events) <= selected or not load_seen(seen_path) <= selected:
        raise ValueError("old session has events or navigation outside its bound selection")
    old_session = make_session(Path(old["scored_csv"]), Path(old["tiles"]), old_state)
    event_path = old_state / "review_labels.csv"
    if event_path.is_file():
        with event_path.open(newline="", encoding="utf-8") as stream:
            for event in csv.DictReader(stream):
                key = key_str(event["image"], event["id"])
                row = old_session._row_for(key)
                if key not in selected or row is None:
                    raise ValueError("old event references an unbound candidate")
                pred, _, _ = compute_final_pred_and_ui(row, old_session.params)
                weights = {"accept":1.0,"flip":1.0,"sus_accept":0.4,"sus_flip":0.4,"skip":0.0}
                if not math.isclose(float(event.get("review_weight", "nan")), weights[event["action"]], abs_tol=1e-9):
                    raise ValueError("old event action weight is invalid")
                expected = pred if event["action"] in {"accept", "sus_accept"} else 1-pred
                if event["action"] == "skip":
                    if event["human_label"] not in {"", "nan", "NaN"}:
                        raise ValueError("old skipped event carries a class label")
                elif event["human_label"] not in {str(expected), str(float(expected))}:
                    raise ValueError("old event label differs from its bound machine decision")
                if int(float(event.get("final_pred", "nan"))) != pred or abs(float(event.get("xgb_prob", "nan"))-float(row["xgb_p"])) > 1e-6:
                    raise ValueError("old event model provenance differs from bound scores")
    make_session(Path(old["scored_csv"]), Path(old["tiles"]), new_state, selection=Path(old["selection"]))
    for name in ("review_labels.csv", "seen.json"):
        if (old_state / name).is_file():
            shutil.copy2(old_state / name, new_state / name)
    receipt = {"schema":"compag-r92-review-expansion/v1", "status":"PASS",
               "source_state":str(old_state),"source_state_sha256":_sha(old_state/"R92_REVIEW_STATE.json"),
               "source_event_sha256":_sha(old_state/"review_labels.csv") if (old_state/"review_labels.csv").is_file() else None,
               "source_seen_sha256":_sha(old_state/"seen.json") if (old_state/"seen.json").is_file() else None,
               "new_state":str(new_state),"candidate_count":len(keys),"prior_accessible_count":len(selected),
               "new_unreviewed_count":len(keys)-len(selected),"preserved_event_keys":len(events)}
    (new_state/"EXPANSION_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def review(scored: Path, tiles: Path, state: Path, *, selection: Path | None = None,
           port: int = 0, open_browser: bool = True) -> int:
    return run_review_web(make_session(scored,tiles,state,selection=selection),port=port,open_browser=open_browser)


def export(state: Path, orig_coco: Path, tiled_coco: Path, output: Path, *, confirm_review_complete: bool) -> dict:
    if not confirm_review_complete:
        raise ValueError("r92 export requires explicit human review completion")
    state = Path(state).resolve(strict=True)
    record = _read_state(state)
    scored = Path(record["scored_csv"])
    rows = _preflight_rows(scored, Path(record["tiles"]))
    keys = {key_str(row["image"],row["id"]) for row in rows}
    labels = load_labels_effective(state/"review_labels.csv")
    manual_actions = {"accept", "flip", "sus_accept", "sus_flip"}
    if record["schema"].endswith("/v2"):
        current = effective_events(state / "review_labels.csv")
        if (set(current) != keys or any(event["action"] not in manual_actions for event in current.values())
                or any(k not in labels or int(float(event["human_label"])) != labels[k]["human_label"]
                       for k, event in current.items())):
            raise ValueError("review is incomplete; every scored candidate needs an effective individual CJ/non-CJ label")
    if len(keys) != len(rows) or any(k not in labels or labels[k]["human_label"] not in (0,1) or labels[k]["action"] not in manual_actions for k in keys):
        raise ValueError("review is incomplete; every scored candidate needs an effective CJ/non-CJ label")
    if set(labels) != keys:
        raise ValueError("review contains decisions outside this score table")
    orig = _registry(Path(orig_coco))
    tiled = _registry(Path(tiled_coco))
    _check_registries(rows, orig, tiled, Path(orig_coco))
    full_binding = record.get("full_inference_receipt")
    if full_binding is not None:
        if (Path(orig_coco).resolve() != Path(full_binding["project"]) / "original_coco.json"
                or Path(tiled_coco).resolve() != Path(full_binding["project"]) / "tiled_coco.json"):
            raise ValueError("export registries must come from the complete inferred project")
    output = Path(output).resolve()
    protected = (Path(record["tiles"]), scored.parent, Path(orig_coco).resolve(), Path(tiled_coco).resolve(), state)
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("export output must be outside protected inputs")
    if output.exists():
        raise FileExistsError("r92 export destination exists; choose a new directory")
    output.mkdir(parents=True)
    try:
        report = merge_reviews(orig_coco_in=Path(orig_coco),tiled_coco_in=Path(tiled_coco),detections_csv=scored,reviews_csv=state/"review_labels.csv",orig_coco_out=Path(output)/"original_coco.json",tiled_coco_out=Path(output)/"tiled_coco.json",new_orig_ids_out=Path(output)/"new_orig_ids.json")
        result = json.loads((Path(output)/"tiled_coco.json").read_text())
        images = {im["file_name"]: im["id"] for im in result["images"]}
        anns = {(ann["image_id"],ann.get("meta",{}).get("review_id")): ann for ann in result["annotations"]}
        for row in rows:
            ann = anns.get((images[row["image"]],int(row["id"])))
            expected_cat = 1 if labels[key_str(row["image"],row["id"])]["human_label"] == 1 else 2
            if ann is None or ann["category_id"] != expected_cat or not ann.get("segmentation"):
                raise ValueError("COCO export did not account for every reviewed candidate")
        if report["reviews_matched"] != len(rows) or report["added"] + report["updated"] != len(rows) or report["skipped"]:
            raise ValueError("COCO export candidate accounting failed")
        if json.loads((Path(output)/"original_coco.json").read_text())["annotations"] != orig["annotations"]:
            raise ValueError("original COCO annotations changed unexpectedly")
        if full_binding is not None and _full_inference_binding(scored, Path(record["tiles"])) != full_binding:
            raise ValueError("full project changed during export")
        receipt={"schema":"compag-r92-review-export/v3","status":"PASS","scored_sha256":record["scored_sha256"],"review_sha256":_sha(state/"review_labels.csv"),"threshold":record["threshold"],"candidate_count":len(rows),"allowed_exclusions":[],"scope":"tiled_coco_new_annotations_only","full_prepared_set":full_binding is not None,"expected_tile_count":full_binding["tile_count"] if full_binding is not None else None,"full_inference_receipt_sha256":full_binding["sha256"] if full_binding is not None else None,"report":report}
        (Path(output)/"EXPORT_RECEIPT.json").write_text(json.dumps(receipt,indent=2,default=str)+"\n")
        hashes = {path.name: _sha(path) for path in Path(output).iterdir() if path.is_file()}
        (Path(output)/"EXPORT_MANIFEST.json").write_text(json.dumps({"schema":"compag-r92-export-manifest/v1","sha256_by_file":dict(sorted(hashes.items()))},indent=2)+"\n")
        return receipt
    except Exception:
        import shutil
        shutil.rmtree(output)
        raise
