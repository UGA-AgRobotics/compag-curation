"""Stage entry points of the ``only-codes-compat-v1`` workflow on a project.

Every entry point reads through the copy-on-write state (originals are
read-only), writes with the legacy relative layout into ``<project>/state``,
snapshots overwritten overlay files and appends a transition record.
The notebook round order is::

    merge(IMG) -> splits(IMG) -> pack -> features(IMG) -> train -> infer(next IMG)
    -> review(next IMG)  [human]  -> next round with IMG = next IMG
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .project import LAYOUT, Project, ProjectError, alloc_next_model_name, next_image_name


def _snapshot(proj: Project, rels: list[str], tag: str) -> list[str]:
    """Copy overlay files that are about to be replaced into snapshots/."""

    saved = []
    snap = proj.root / "snapshots" / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{tag}"
    for rel in rels:
        ov = proj.state.overlay(rel)
        if ov.exists() and ov.is_file():
            dst = snap / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ov, dst)
            saved.append(rel)
    return saved


def image_rel(image: str) -> str:
    return f"{LAYOUT['maskout_tile_root']}/{image}"


def infer(proj: Project, image: str, *, model_round_dir: str, device: str = "cuda", mode: str = "xgb_recall",
          xgb_threshold: str = "0.50", det_missing: str = "reject", execution_variant: dict | None = None,
          replace: bool = False, detector: str = "required") -> dict[str, Any]:
    """Reference tiled inference for one image.

    ``detector="required"`` (the reference hybrid preset) fails in preflight when the YOLO model is
    absent instead of silently running the detector-free branch; ``detector="none"`` selects that
    branch explicitly and records it as a declared variant.
    """

    from .inference import run_legacy_inference

    st = proj.state
    tiles = st.path(f"{image_rel(image)}/tiles")
    if not tiles.is_dir():
        raise ProjectError(f"tiles not found for {image}: {tiles}")
    run_root = st.overlay(image_rel(image))
    run_dir = run_root / f"run_{mode}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not replace:
            raise ProjectError(f"{run_dir} exists; pass --replace to snapshot and recompute")
        snap = proj.root / "snapshots" / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_infer_{image}"
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(snap))
    proto = st.path(LAYOUT["infer_proto_padded"])
    if not proto.exists():
        # runner: create the padded prototype from the unpadded one when absent
        import pandas as pd

        src = st.path(LAYOUT["infer_proto"])
        df = pd.read_csv(src); ren = {}
        for c in df.columns:
            if c.startswith("mu_embed_"):
                try:
                    ren[c] = f"mu_embed_{int(c.split('_')[-1]):04d}"
                except Exception:
                    pass
        proto = st.write_path(LAYOUT["infer_proto_padded"])
        df.rename(columns=ren).to_csv(proto, index=False)
    yolo = st.path(LAYOUT["yolo_weights"])
    pinned = {}
    for row in proj.manifest.get("inventory", []):
        if row.get("input_type", "").startswith("YOLO11n") and row.get("sha256_or_tree"):
            pinned["yolo_weights"] = row["sha256_or_tree"]
        if row.get("input_type", "").startswith("SAM2.1") and row.get("sha256_or_tree"):
            pinned["sam2_ckpt"] = row["sha256_or_tree"]
        if row.get("input_type", "").startswith("ResNet50") and row.get("sha256_or_tree"):
            pinned["resnet50_weights"] = row["sha256_or_tree"]
        if row.get("input_type", "").startswith("inference prototype (padded)") and row.get("sha256_or_tree"):
            pinned["padded_proto"] = row["sha256_or_tree"]
    manifest = run_legacy_inference(
        tiles_dir=tiles, run_root=run_root, model_bundle_dir=proj.bundle_for(model_round_dir),
        padded_proto=proto, pca_npz=st.path(LAYOUT["infer_pca"]), sam2_ckpt=proj.sam2_ckpt(),
        resnet50_weights=proj.resnet50(), yolo_weights=yolo,
        sam2_config=proj.manifest["sam2_config"], mode=mode, xgb_threshold=xgb_threshold, device=device,
        det_missing=det_missing, execution_variant=execution_variant, detector=detector,
        pinned_inputs=pinned or None)
    proj.record_transition("infer", {
        "image": image, "model": model_round_dir,
        "reference_preset_requested": manifest["reference_preset_requested"],
        "effective_configuration_match": manifest["effective_configuration_match"]["status"],
        "inputs_verified": manifest["inputs_verified"]["status"],
        "software_environment_match": manifest["software_environment_match"].get("software_environment_match"),
        "hardware_match": manifest["software_environment_match"].get("hardware_match"),
        "detector": detector, "execution_variant": manifest["execution_variant"],
        "run_completed": manifest["run_completed"],
        "output_parity_verification": manifest["output_parity_verification"]["status"],
        "outputs": len(manifest.get("outputs", {}))})
    return manifest


def _mapped_dir(proj: Project, path: Path) -> Path:
    """Apply the project's explicit relocation rules to a directory the reviewer reads.

    The GUI uses the same rules as the pack/feature readers: the recorded location wins when it
    exists, otherwise an explicit rule may point at the relocated copy.
    """

    if path.exists():
        return path
    mapped = Path(proj.mapper.map(str(path)))
    return mapped if mapped.exists() else path


def ensure_review_state(proj: Project, image: str, *, mode: str = "xgb_recall",
                        review_labels_source: str | None = None) -> dict[str, Path]:
    """Bring historical review state for ``image`` into the overlay (copy, never move)."""

    st = proj.state
    base = image_rel(image)
    det_rel = f"{base}/run_{mode}/detections.csv"
    if not st.exists(det_rel):
        for m in ("xgb_recall", "xgb", "gate"):
            cand = f"{base}/run_{m}/detections.csv"
            if st.exists(cand):
                det_rel = cand
                mode = m
                break
    rev_rel = f"{base}/review_labels.csv"
    if review_labels_source:
        src = Path(review_labels_source)
        dst = st.overlay(rev_rel)
        if dst.exists():
            raise ProjectError(f"{rev_rel} already exists in the project; refusing to replace review history")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        proj.record_transition("review-import", {"image": image, "source": str(src)})
    elif not st.overlay(rev_rel).exists() and st.exists(rev_rel):
        st.ensure_overlay_copy(rev_rel)
    seen_rel = f"{base}/run_{mode}/_dash_cache/seen.json"
    if not st.overlay(seen_rel).exists() and st.exists(seen_rel):
        st.ensure_overlay_copy(seen_rel)
    return {
        "detections": st.path(det_rel),
        "tiles": _mapped_dir(proj, st.path(f"{base}/tiles")),
        "review_csv": st.overlay(rev_rel),
        "seen_json": st.overlay(seen_rel),
        "cache_dir": st.overlay(f"{base}/run_{mode}/_dash_cache"),
    }


def open_review_session(proj: Project, image: str, **kw):
    from .review import LegacyReviewSession

    paths = ensure_review_state(proj, image, **kw)
    return LegacyReviewSession(detections_csv=paths["detections"], tiles_dir=paths["tiles"],
                               review_csv=paths["review_csv"], seen_json=paths["seen_json"],
                               cache_dir=paths["cache_dir"])


def merge(proj: Project, image: str, *, mode: str = "xgb_recall", confirm_review_complete: bool) -> dict[str, Any]:
    from .merge import merge_reviews

    if not confirm_review_complete:
        raise ProjectError("merge requires explicit human approval (--confirm-review-complete)")
    st = proj.state
    base = image_rel(image)
    rev = st.path(f"{base}/review_labels.csv")
    det = _resolve_detections_in_view(proj, base, mode)
    outs = [LAYOUT["orig_coco"], LAYOUT["tiled_coco"], f"{base}/round_k_new_orig_ids.json"]
    saved = _snapshot(proj, outs, f"merge_{image}")
    report = merge_reviews(orig_coco_in=st.path(LAYOUT["orig_coco"]), tiled_coco_in=st.path(LAYOUT["tiled_coco"]),
                           detections_csv=det, reviews_csv=rev,
                           orig_coco_out=st.write_path(LAYOUT["orig_coco"]),
                           tiled_coco_out=st.write_path(LAYOUT["tiled_coco"]),
                           new_orig_ids_out=st.write_path(f"{base}/round_k_new_orig_ids.json"))
    proj.record_transition("merge", {"image": image, "detections": str(det), "reviews": str(rev),
                                     "snapshotted": saved, **{k: v for k, v in report.items() if k != "new_orig_ids"}})
    return report


def _resolve_detections_in_view(proj: Project, base: str, mode: str) -> Path:
    """merge.resolve_detections_csv semantics over the copy-on-write view."""

    st = proj.state
    mode = (mode or "xgb_recall").strip().lower()
    if mode not in {"gate", "xgb", "xgb_recall"}:
        mode = "xgb_recall"
    rel = f"{base}/run_{mode}/detections.csv"
    if not st.exists(rel):
        for m in ("xgb_recall", "xgb", "gate"):
            cand = f"{base}/run_{m}/detections.csv"
            if st.exists(cand):
                rel = cand
                break
    return st.path(rel)


def splits(proj: Project, image: str) -> dict[str, Any]:
    from .splits import update_splits

    st = proj.state
    split_rel = LAYOUT["split_dir"]
    for name in ("orig_train_ids.json", "orig_val_ids.json", "orig_test_ids.json"):
        if not st.overlay(f"{split_rel}/{name}").exists() and st.exists(f"{split_rel}/{name}"):
            st.ensure_overlay_copy(f"{split_rel}/{name}")
    append_rel = f"{image_rel(image)}/round_k_new_orig_ids.json"
    if not st.overlay(append_rel).exists() and st.exists(append_rel):
        st.ensure_overlay_copy(append_rel)
    rep = update_splits(orig_coco_json=st.path(LAYOUT["orig_coco"]), split_dir=st.overlay(split_rel),
                        append_new_orig_ids=st.overlay(append_rel))
    proj.record_transition("splits", {"image": image, **rep})
    return rep


def pack(proj: Project, *, device: str = "cuda", run_mode: str = "all") -> dict[str, Any]:
    from .pack import run_pack

    st = proj.state
    rep = run_pack(state=st, coco_json=st.path(LAYOUT["tiled_coco"]), images_dir=st.path(LAYOUT["tiles_dir"]),
                   art_rel=LAYOUT["art_root"], split_rel=LAYOUT["split_dir"], fold_tag=proj.manifest["fold_tag"],
                   resnet50_weights=str(proj.resnet50()), device=device, run_mode=run_mode,
                   pos_name=proj.manifest["pos_name"], mapper=proj.mapper)
    proj.record_transition("pack", rep)
    return rep


def features(proj: Project, image: str, *, device: str = "cuda") -> dict[str, Any]:
    from .features import run_feature_export
    from .inference import runtime_versions

    st = proj.state
    pro = f"{LAYOUT['art_root']}/stage2_proto/{proj.manifest['fold_tag']}"
    rep = run_feature_export(
        state=st, coco_json=st.path(LAYOUT["tiled_coco"]), images_dir=st.path(LAYOUT["tiles_dir"]),
        pack_path=st.path(f"{pro}/foldsafe_pack.joblib"), out_csv_rel=LAYOUT["features_csv"],
        allowed_img_ids_json=st.path(f"{pro}/used_img_ids.json"),
        review_csv=st.path(f"{image_rel(image)}/review_labels.csv"), resnet50_weights=str(proj.resnet50()),
        device=device, pos_name=proj.manifest["pos_name"], mapper=proj.mapper,
        transform_identity={"profile": "only-codes-compat-v1", "scales": [0.67, 0.8, 1.0, 1.25],
                            "pack_sha256": _sha(st.path(f"{pro}/foldsafe_pack.joblib")),
                            "resnet50_sha256": _sha(proj.resnet50()), "versions": runtime_versions()})
    proj.record_transition("features", {"image": image, "appended_rows": rep["appended_rows"],
                                        "migrations": rep["migrations"]})
    return rep


def _sha(p: Path) -> str | None:
    from .state import sha256_file

    return sha256_file(p) if Path(p).exists() else None


def train(proj: Project, *, device: str = "cuda", model_name: str | None = None, round_dir: str | None = None,
          log=print) -> dict[str, Any]:
    """Cell3A+3B on the project's feature table; allocates the next round like the bootstrap."""

    from .model_bundle import bundle_from_object
    from .training import (fit_and_evaluate, load_prev_best_params, prepare_training_data,
                           read_features_csv_robust, write_round_artifacts, xgb_setup)

    st = proj.state
    mr = LAYOUT["model_root"]
    if model_name is None or round_dir is None:
        model_name, round_dir = alloc_next_model_name([st.original(mr), st.overlay(mr)],
                                                      proj.manifest.get("bootstrap_model_name"))
    m_r = re.search(r"r(\d+)(?:\D|$)", str(round_dir) or str(model_name), flags=re.IGNORECASE)
    round_tag = f"r{m_r.group(1)}" if m_r else "r1"
    csv_path = st.path(LAYOUT["features_csv"])
    df, used = read_features_csv_robust(csv_path, st.write_path(LAYOUT["features_csv"].replace(".csv", ".normalized.csv")))
    legacy_csv_path = proj.mapper.map(str(Path(proj.manifest["roots"]["legacy_root"]) / LAYOUT["features_csv"]))
    prep = prepare_training_data(df=df, csv_used=legacy_csv_path if used == csv_path else str(used), state=st,
                                 model_root_rel=mr, round_tag=round_tag, round_dir_name=round_dir, log=log)
    setup = xgb_setup(device)
    prev_best, prev_best_dir = load_prev_best_params(st, mr, round_tag, round_dir)
    fit = fit_and_evaluate(prep, setup, round_tag=round_tag, model_name=model_name, prev_best_params=prev_best, log=log)
    lineage = {"features_csv": str(csv_path), "features_csv_sha256": _sha(csv_path), "prev_best_params_dir": prev_best_dir,
               "prev_round": prep["prev"], "device": device, "xgb_setup": {k: v for k, v in setup.items() if k != "xgb_kwargs"}}
    art = write_round_artifacts(state=st, model_root_rel=mr, round_dir_name=round_dir, round_tag=round_tag,
                                model_name=model_name, prep=prep, fit=fit, csv_path_recorded=legacy_csv_path,
                                lineage=lineage, log=log)
    import pandas as pd

    vr = prep["X_test"].head(2000) if len(prep["X_test"]) else prep["X_train"].head(2000)
    obj = {"pipeline": fit["best_model"], "features": list(prep["num_cols"]), "threshold": float(fit["thr_use"]),
           "meta": art["meta"]}
    bundle_rel = f"bundles/{round_dir}"
    bman = bundle_from_object(obj, proj.root / bundle_rel,
                              source={"path": art["model_path"], "sha256": _sha(Path(art["model_path"])),
                                      "trust": "TRAINED_BY_THIS_APPLICATION", "format": "in-memory",
                                      "model_id": art["identity"]["model_id"]},
                              verification_rows=pd.DataFrame(vr, columns=prep["num_cols"]))
    proj.manifest["model_bundles"][round_dir] = {"bundle_dir": bundle_rel, "source": bman["source"],
                                                 "verification": bman["verification"]}
    proj.save()
    proj.record_transition("train", {"round_dir": round_dir, "model_name": model_name, "model_id": art["identity"]["model_id"],
                                     "used_trees": fit["used_trees"], "threshold": fit["thr_use"],
                                     "param_source": fit["param_source"]})
    return {"round_dir": round_dir, "model_name": model_name, "round_tag": round_tag, "artifacts": art,
            "used_trees": fit["used_trees"], "threshold": fit["thr_use"], "param_source": fit["param_source"],
            "tile_pr_auc": fit["tile_pr_auc"]}


def round_after_review(proj: Project, image: str, *, device: str = "cuda", confirm_review_complete: bool,
                       run_inference: bool = True, log=print) -> dict[str, Any]:
    """The notebook cell sequence for one round, stopping before the next human review."""

    out: dict[str, Any] = {"image": image}
    out["merge"] = merge(proj, image, confirm_review_complete=confirm_review_complete)
    out["splits"] = splits(proj, image)
    out["pack"] = pack(proj, device=device)
    out["features"] = features(proj, image, device=device)
    out["train"] = train(proj, device=device, log=log)
    st = proj.state
    nxt = next_image_name([st.original(LAYOUT["maskout_tile_root"]), st.overlay(LAYOUT["maskout_tile_root"])], image)
    out["next_image"] = nxt
    if run_inference:
        out["infer"] = infer(proj, nxt, model_round_dir=out["train"]["round_dir"], device=device,
                             replace=(nxt == image))
    proj.manifest["current_image"] = nxt
    proj.save()
    return out


def readiness(proj: Project, image: str | None = None) -> dict[str, Any]:
    """Per-entry-point input readiness (reported, never auto-substituted)."""

    st = proj.state
    image = image or proj.manifest.get("current_image")
    base = image_rel(image) if image else None
    has = lambda rel: bool(rel) and st.exists(rel)  # noqa: E731
    det = None
    if base:
        det = next((f"{base}/run_{m}/detections.csv" for m in ("xgb_recall", "xgb", "gate") if has(f"{base}/run_{m}/detections.csv")), None)
    out = {
        "predict": {"ready": bool(base and has(f"{base}/tiles") and proj.manifest.get("model_bundles")
                                  and (has(LAYOUT["infer_proto_padded"]) or has(LAYOUT["infer_proto"])) and has(LAYOUT["infer_pca"])
                                  and proj.sam2_ckpt().exists() and proj.resnet50().exists() and has(LAYOUT["yolo_weights"])),
                    "needs": "tiles, a verified model bundle, inference prototype + PCA, SAM2 Hiera-L, ResNet50, YOLO"},
        "review": {"ready": bool(det and has(f"{base}/tiles")), "detections": det},
        "merge": {"ready": bool(det and has(f"{base}/review_labels.csv") and has(LAYOUT["orig_coco"]) and has(LAYOUT["tiled_coco"])),
                  "review_labels": (f"{base}/review_labels.csv" if base else None),
                  "note": ("review_labels.csv is absent for this image; other logs (e.g. review_labels01.csv) are never "
                           "substituted automatically -- import one explicitly with 'review --review-labels'"
                           if base and not has(f"{base}/review_labels.csv") else "")},
        "splits": {"ready": has(LAYOUT["orig_coco"]) and has(f"{LAYOUT['split_dir']}/orig_train_ids.json")},
        "pack": {"ready": has(LAYOUT["tiled_coco"]) and has(LAYOUT["tiles_dir"]) and proj.resnet50().exists()},
        "features": {"ready": has(LAYOUT["tiled_coco"]) and has(f"{LAYOUT['art_root']}/stage2_proto/{proj.manifest['fold_tag']}/foldsafe_pack.joblib")},
        "train": {"ready": has(LAYOUT["features_csv"])},
    }
    return {"image": image, "entry_points": out}
