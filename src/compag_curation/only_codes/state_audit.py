"""Inspection (and bounded, opt-in repair) of a project overlay written before the 1.9.1 fix.

v1.9.0 committed an image id to ``used_img_ids`` even when its tile image could not be read, so a
later resume reported ``SKIP_NO_NEW_TRAIN_IMAGES`` and the missing embeddings were never built.
This module reports what that overlay can and cannot prove, never rewrites a historical pack or
feature table in place, and offers a rebuild into a *new* overlay when the evidence is ambiguous.

Runs read-only unless ``rebuild_into`` is given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import resolver_for
from .project import LAYOUT, Project

SCHEMA = "compag-only-codes-state-audit/v1"


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def inspect_state(proj: Project, *, fold_tag: str | None = None) -> dict[str, Any]:
    import numpy as np

    st = proj.state
    fold = fold_tag or proj.manifest.get("fold_tag", "train_foldA")
    art = LAYOUT["art_root"]
    emb = f"{art}/stage1_embeddings/{fold}"
    pro = f"{art}/stage2_proto/{fold}"
    rep: dict[str, Any] = {"schema": SCHEMA, "project": str(proj.root), "fold_tag": fold,
                           "overlay_written": {}, "findings": [], "status": "CLEAN"}

    receipts = _load_json(st.overlay(f"{pro}/stage1_receipts.json"))
    used = _load_json(st.path(f"{pro}/used_img_ids.json")) or []
    used = [int(x) for x in used] if isinstance(used, list) else []
    pos = st.path(f"{emb}/embeddings_train_pos.npy")
    all_ = st.path(f"{emb}/embeddings_train_all.npy")
    n_pos = int(np.load(pos, mmap_mode="r").shape[0]) if pos.exists() else 0
    n_all = int(np.load(all_, mmap_mode="r").shape[0]) if all_.exists() else 0
    rep["overlay_written"] = {k: st.source_of(k) == "PROJECT_OVERLAY" for k in
                              (f"{emb}/embeddings_train_pos.npy", f"{emb}/embeddings_train_all.npy",
                               f"{pro}/jassid_unified_proto.csv", f"{pro}/foldsafe_pack.joblib",
                               f"{pro}/used_img_ids.json")}
    rep["counts"] = {"used_img_ids": len(used), "embeddings_pos_rows": n_pos, "embeddings_all_rows": n_all,
                     "receipt_runs": len(receipts or [])}

    # 1) which used ids are backed by a receipt (1.9.1 onwards) and which are unexplained
    receipted: dict[int, str] = {}
    for run in receipts or []:
        for k, v in (run.get("images") or {}).items():
            receipted[int(k)] = v.get("outcome", "?")
    unexplained = [i for i in used if i not in receipted]
    rep["receipted_outcomes"] = {str(k): v for k, v in sorted(receipted.items())}
    rep["used_ids_without_receipt"] = unexplained

    # 2) can every used id still be read today?  (same resolver the pack reader uses)
    coco = st.path(LAYOUT["tiled_coco"])
    unreadable: list[dict[str, Any]] = []
    if coco.exists():
        data = json.loads(coco.read_text(encoding="utf-8"))
        id_to_img = {int(im["id"]): im for im in data.get("images", [])}
        with_ann = {int(a.get("image_id", -1)) for a in data.get("annotations", [])}
        resolver = resolver_for("pack", st.path(LAYOUT["tiles_dir"]), proj.mapper, record_identity=False)
        for i in used:
            info = id_to_img.get(int(i))
            if info is None:
                unreadable.append({"image_id": i, "reason": "IMAGE_ID_NOT_IN_TILED_COCO"})
                continue
            r = resolver.resolve(str(info.get("file_name", "")))
            if not r.ok:
                unreadable.append({"image_id": i, "reason": r.outcome, "file_name": str(info.get("file_name", "")),
                                   "candidates": r.candidates,
                                   "has_annotations": int(i) in with_ann,
                                   "receipt": receipted.get(int(i), "NONE")})
    rep["used_ids_unreadable_today"] = unreadable

    if unreadable:
        rep["findings"].append({
            "id": "USED_ID_UNREADABLE",
            "detail": f"{len(unreadable)} image id(s) are marked as used but cannot be read with the current "
                      "roots and mapping rules; if they were committed by the 1.9.0 defect their embeddings "
                      "are missing from the pack",
            "ambiguity": "the 1.9.0 overlay has no per-image receipt, so a missing contribution cannot be "
                         "distinguished from a legitimate zero-contribution image by inspection alone",
            "remedy": "supply the correct --map rule and rebuild the derived scope into a NEW overlay "
                      "(--rebuild-into DIR); the historical pack and feature table are never rewritten"})
        rep["status"] = "AMBIGUOUS_OVERLAY"
    elif unexplained and (rep["overlay_written"].get(f"{pro}/foldsafe_pack.joblib")):
        rep["findings"].append({
            "id": "NO_RECEIPTS",
            "detail": f"{len(unexplained)} used image id(s) were committed by a pre-1.9.1 run with no receipt",
            "ambiguity": "every id is readable today, so nothing is known to be missing, but the overlay "
                         "cannot prove which ids actually contributed vectors",
            "remedy": "re-run pack in a new overlay if a proof of contribution is required"})
        rep["status"] = "UNPROVEN_BUT_READABLE"
    return rep


def rebuild_into(proj: Project, target: Path, *, device: str = "cuda", fold_tag: str | None = None) -> dict[str, Any]:
    """Rebuild only the derived TRAIN scope (embeddings, prototype, PCA, pack) into a new project.

    The source project is opened read-only: its original workspace stays the state's original root,
    and everything is written into ``target``.  Nothing of the affected overlay is modified.
    """

    from .pack import run_pack
    from .project import open_project
    from .state import LegacyState

    target = Path(target)
    if target.exists():
        raise FileExistsError(f"rebuild target already exists: {target}")
    target.mkdir(parents=True)
    src_state = proj.state
    st = LegacyState(src_state.original_root, target / "state")
    rep = run_pack(state=st, coco_json=src_state.path(LAYOUT["tiled_coco"]),
                   images_dir=src_state.path(LAYOUT["tiles_dir"]), art_rel=LAYOUT["art_root"],
                   split_rel=LAYOUT["split_dir"], fold_tag=fold_tag or proj.manifest.get("fold_tag", "train_foldA"),
                   resnet50_weights=str(proj.resnet50()), device=device, pos_name=proj.manifest["pos_name"],
                   mapper=proj.mapper)
    out = {"schema": SCHEMA, "rebuilt_into": str(target), "source_project": str(proj.root), "pack_report": rep,
           "note": "derived TRAIN scope only; review logs, COCO masters and the historical feature table "
                   "were not copied or modified"}
    (target / "REBUILD_REPORT.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str),
                                                encoding="utf-8")
    return out
