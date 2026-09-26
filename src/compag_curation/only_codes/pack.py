"""Port of notebook cell 4 "JUPYTER CELL 1 — TRAIN-only pack (mu from POS, PCA from ALL)".

Incremental Stage-1 embeddings (positives for mu, all annotations for PCA)
over TRAIN tiled images, Stage-2 prototype CSV, Stage-3 PCA and the fold-safe
pack.  Scales are [0.85, 1.00, 1.15] applied to the *masked patch* (crop
first, then resize).  All reads resolve through the copy-on-write state; all
writes go to the project overlay with the original relative layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .inputs import InputResolver, resolver_for
from .state import LegacyState, PathMapper

SCALES_TRAIN = [0.85, 1.00, 1.15]


class PackInputError(RuntimeError):
    """Raised when an input of this round could not be read.

    The round is aborted before anything is published: no embeddings are appended, no image id is
    marked as used and the previous pack/prototype/PCA stay exactly as they were.  ``report``
    carries the per-image outcomes and resolution records.
    """

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report = report


def _l2norm(a: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(a, axis=axis, keepdims=True) + eps
    return a / n


def _seg_to_mask(ann: Dict[str, Any], H: int, W: int) -> np.ndarray:
    import cv2

    m = np.zeros((H, W), dtype=np.uint8)
    seg = ann.get("segmentation", None)
    if isinstance(seg, list) and seg:
        polys = []
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            polys.append(pts.astype(np.int32))
        cv2.fillPoly(m, polys, 255)
        return m
    if isinstance(seg, dict) and "counts" in seg:
        try:
            import pycocotools.mask as maskUtils
            rle = seg if isinstance(seg.get("counts"), (list, str, bytes)) else maskUtils.frPyObjects(seg, H, W)
            mm = maskUtils.decode(rle)
            if mm.ndim == 3:
                mm = mm.max(axis=2)
            return (mm.astype(np.uint8) * 255)
        except Exception:
            pass
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W - 1, int(x + w)), min(H - 1, int(y + h))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


def _join_image_path(images_root: Path, file_name_value: str) -> Path:
    """Reference join of cell 3 (kept verbatim; used when no relocation rule applies)."""

    fn = Path(str(file_name_value))
    return fn if fn.is_absolute() else (images_root / fn)


def map_orig_to_tiled_ids(coco_json_path: Path, orig_ids: Iterable[int]) -> List[int]:
    data = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    imgs = data.get("images", [])
    want = set(int(x) for x in orig_ids)
    out = []
    for im in imgs:
        oid = im.get("meta", {}).get("orig_image_id", None)
        if oid is None:
            continue
        if int(oid) in want:
            out.append(int(im["id"]))
    return sorted(out)


class _Embedder:
    """Model + transform with the notebook's robust per-patch CPU fallback."""

    def __init__(self, weights_path: str, device: str):
        from .legacy.gate_core import build_embed_model

        model, tf = build_embed_model(weights_path)
        self.fallback_count = 0
        try:
            self.model = model.to(device).eval()
            self.device = device
        except Exception:
            self.model = model.to("cpu").eval()
            self.device = "cpu"
            self.fallback_count += 1
        self.tf = tf

    def embed(self, patch):
        from .legacy.gate_core import embed_patch

        try:
            return embed_patch(patch, self.model, self.tf, device=self.device).astype(np.float32)
        except Exception:
            self.fallback_count += 1
            try:
                return embed_patch(patch, self.model.to("cpu").eval(), self.tf, device="cpu").astype(np.float32)
            except Exception:
                return None


def build_embeddings_dual(*, coco_json: Path, images_dir: Path, save_pos_npy: Path, save_all_npy: Path,
                          include_img_ids: Set[int], target_cat_names: Set[str], embedder: _Embedder,
                          sample_limit: Optional[int] = None, pad_frac: float = 0.10,
                          resolver: Optional[InputResolver] = None) -> Dict[str, Any]:
    """Cell-3 embedding build.

    The scientific loop (annotation order, scales, patch, embedding) is the reference one.  What is
    added: every image path is resolved through :class:`InputResolver` (identical to the reference
    join when no relocation rule applies) and each image gets an explicit outcome, so the caller can
    tell a hard read/model failure from a legitimate zero contribution.
    """

    import cv2

    from .legacy.gate_core import crop_masked_patch, yellow_bg_stats

    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    id_to_img = {int(im["id"]): im for im in data.get("images", [])}
    id_to_cat = {int(c["id"]): c.get("name", "") for c in data.get("categories", [])}
    target_lc = {s.lower() for s in target_cat_names}
    pos_cat_ids = {cid for cid, nm in id_to_cat.items() if (nm or "").lower() in target_lc}
    resolver = resolver or resolver_for("pack", images_dir, None)

    vecs_pos: List[np.ndarray] = []
    vecs_all: List[np.ndarray] = []
    kept_total = 0
    D_first: Optional[int] = None
    unreadable: list[str] = []
    images: Dict[int, Dict[str, Any]] = {}
    resolved_cache: Dict[int, Any] = {}

    def _img_rec(img_id: int, info: Dict[str, Any]) -> Dict[str, Any]:
        rec = images.get(img_id)
        if rec is None:
            rec = {"file_name": str(info.get("file_name", "")), "outcome": "NO_ELIGIBLE_INSTANCES",
                   "vectors": 0, "annotations_seen": 0}
            images[img_id] = rec
        return rec

    for ann in data.get("annotations", []):
        img_id = int(ann.get("image_id", -1))
        if img_id not in include_img_ids:
            continue
        is_pos = (ann.get("category_id") in pos_cat_ids)
        info = id_to_img.get(img_id)
        if not info:
            continue
        rec = _img_rec(img_id, info)
        rec["annotations_seen"] += 1
        if img_id not in resolved_cache:
            resolved_cache[img_id] = resolver.resolve(str(info.get("file_name", "")))
        resolution = resolved_cache[img_id]
        rec.setdefault("resolution", resolution.to_dict())
        img_path = resolution.path      # single authoritative reader-aware resolution
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if resolution.ok else None
        if img is None:
            unreadable.append(str(img_path) if img_path else resolution.logical)
            rec["outcome"] = "READ_FAILED"
            rec["failure"] = {"path": str(img_path) if img_path else None,
                              "resolution_outcome": resolution.outcome,
                              "reference_candidate": resolution.reference_candidate}
            continue
        H, W = img.shape[:2]
        mask = _seg_to_mask(ann, H, W)
        _, _, bg_bgr = yellow_bg_stats(img)
        patch = crop_masked_patch(img, mask, pad_frac=pad_frac, bg_bgr=bg_bgr)
        if patch is None:
            if rec["outcome"] == "NO_ELIGIBLE_INSTANCES":
                rec["outcome"] = "NO_USABLE_PATCH"
            continue
        for s in SCALES_TRAIN:
            if abs(s - 1.0) < 1e-4:
                patch_s = patch
            else:
                h, w = patch.shape[:2]
                patch_s = cv2.resize(patch, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
            z = embedder.embed(patch_s)
            if z is None:
                rec["outcome"] = "EMBED_FAILED"
                rec["failure"] = {"path": str(img_path), "scale": s, "reason": "embedding model returned nothing"}
                continue
            z = _l2norm(z).astype(np.float32)
            if D_first is None:
                D_first = int(z.shape[-1])
            vecs_all.append(z)
            if is_pos:
                vecs_pos.append(z)
            kept_total += 1
            rec["vectors"] += 1
            if rec["outcome"] in ("NO_ELIGIBLE_INSTANCES", "NO_USABLE_PATCH"):
                rec["outcome"] = "CONTRIBUTED"
            if sample_limit and kept_total >= sample_limit:
                break
        if sample_limit and kept_total >= sample_limit:
            break

    D_default = int(D_first) if D_first is not None else 2048
    Z_pos = (np.stack(vecs_pos, axis=0).astype(np.float32) if vecs_pos else np.zeros((0, D_default), np.float32))
    Z_all = (np.stack(vecs_all, axis=0).astype(np.float32) if vecs_all else np.zeros((0, D_default), np.float32))
    save_pos_npy.parent.mkdir(parents=True, exist_ok=True)
    save_all_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_pos_npy, Z_pos)
    np.save(save_all_npy, Z_all)
    for img_id in sorted(include_img_ids):
        if img_id not in images:
            info = id_to_img.get(int(img_id))
            images[int(img_id)] = {"file_name": str((info or {}).get("file_name", "")),
                                   "outcome": "NO_ANNOTATIONS_IN_SCOPE", "vectors": 0, "annotations_seen": 0}
    failed = sorted(i for i, r in images.items() if r["outcome"] in ("READ_FAILED", "EMBED_FAILED"))
    contributed = sorted(i for i, r in images.items() if r["outcome"] == "CONTRIBUTED")
    zero = sorted(i for i, r in images.items()
                  if r["outcome"] in ("NO_ELIGIBLE_INSTANCES", "NO_USABLE_PATCH", "NO_ANNOTATIONS_IN_SCOPE"))
    return {"pos_shape": list(Z_pos.shape), "all_shape": list(Z_all.shape), "unreadable": unreadable,
            "images": {str(k): v for k, v in sorted(images.items())},
            "failed_ids": failed, "contributed_ids": contributed, "zero_contribution_ids": zero,
            "input_resolution": resolver.summary()}


def make_unified_proto_from_embeddings(emb_pos_npy: Path, proto_csv: Path, thresh_method: str = "fixed",
                                       fixed_thresh: float = 0.70, percentile: float = 5.0):
    import pandas as pd

    Z = np.load(emb_pos_npy).astype(np.float32)
    if Z.ndim != 2 or Z.shape[0] < 1:
        return None, None, None
    Z = _l2norm(Z, axis=1)
    mu = _l2norm(Z.mean(axis=0, keepdims=False))[...]
    sims = (Z @ mu)
    embed_thresh = float(np.percentile(sims, float(percentile))) if thresh_method == "percentile" else float(fixed_thresh)
    cols = [f"mu_embed_{i}" for i in range(mu.shape[0])] + ["embed_thresh"]
    vals = list(mu.tolist()) + [embed_thresh]
    df = pd.DataFrame([vals], columns=cols)
    proto_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proto_csv, index=False)
    return proto_csv, float(embed_thresh), int(mu.shape[0])


def fit_pca_from_embeddings(emb_all_npy: Path, out_npz: Path, k: int = 32):
    Z = np.load(emb_all_npy).astype(np.float32)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return None, None, None
    mu = Z.mean(axis=0, keepdims=True); X = Z - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:k].astype(np.float32)
    mean = mu.squeeze(0).astype(np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, components=components, mean=mean)
    return out_npz, (int(components.shape[0]), int(components.shape[1])), int(mean.shape[0])


def _load_prev_used_ids(pack_path: Path) -> Set[int]:
    from .safe_joblib import load_data_pack

    if not pack_path.exists():
        return set()
    try:
        pack = load_data_pack(pack_path)
        return set(map(int, pack.get("used_img_ids", [])))
    except Exception:
        return set()


def _append_receipts(state: LegacyState, rel: str, record: Dict[str, Any]) -> Path:
    """Append one stage-1 receipt to the project overlay (never to the original workspace).

    The receipts make a later inspection unambiguous: which image ids actually contributed
    vectors, which were a legitimate zero contribution, which failed, and from which resolved
    path each one was read.
    """

    import time

    p = state.overlay(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        if not isinstance(existing, list):
            existing = [existing]
    except Exception:
        existing = []
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    existing.append(rec)
    _atomic_write(p, lambda tmp: tmp.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str),
                                                encoding="utf-8"))
    return p


def _atomic_write(target: Path, writer) -> Path:
    """Write through a temporary file in the same directory and replace in one step."""

    target.parent.mkdir(parents=True, exist_ok=True)
    # keep the original suffix last: numpy appends ".npy" to a path that does not end with it
    tmp = target.with_name(target.name + ".publishing" + target.suffix)
    try:
        writer(tmp)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def run_pack(*, state: LegacyState, coco_json: Path, images_dir: Path, art_rel: str, split_rel: str,
             fold_tag: str, resnet50_weights: str, device: str, run_mode: str = "all", pos_name: str = "cj",
             sample_limit: Optional[int] = None, pad_frac: float = 0.10, pca_dims: int = 32,
             thresh_method: str = "fixed", fixed_thresh: float = 0.70, percentile: float = 5.0,
             mapper: Optional[PathMapper] = None, embedder_factory=None) -> Dict[str, Any]:
    import joblib
    import pandas as pd

    run_mode = run_mode.lower()
    if run_mode == "skip":
        return {"mode": "skip"}
    emb = f"{art_rel}/stage1_embeddings/{fold_tag}"
    pro = f"{art_rel}/stage2_proto/{fold_tag}"
    pca = f"{art_rel}/stage3_pca/{fold_tag}"
    EMB_POS = f"{emb}/embeddings_train_pos.npy"
    EMB_ALL = f"{emb}/embeddings_train_all.npy"
    PROTO = f"{pro}/jassid_unified_proto.csv"
    PCA = f"{pca}/embed_pca_32.npz"  # the notebook hard-codes this name for any CJ_PCA_DIMS
    PACK = f"{pro}/foldsafe_pack.joblib"
    USED = f"{pro}/used_img_ids.json"
    USED_TRAIN = f"{split_rel}/used_img_ids_train.json"
    ORIG_TRAIN = f"{split_rel}/orig_train_ids.json"

    if state.exists(USED_TRAIN):
        include = set(map(int, json.loads(state.path(USED_TRAIN).read_text(encoding="utf-8"))))
        split_source = "used_img_ids_train.json"
    elif state.exists(ORIG_TRAIN):
        orig_ids = list(map(int, json.loads(state.path(ORIG_TRAIN).read_text(encoding="utf-8"))))
        include = set(map_orig_to_tiled_ids(coco_json, orig_ids))
        split_source = "orig_train_ids.json -> meta.orig_image_id"
    else:
        raise RuntimeError("No TRAIN subset provided (used_img_ids_train.json or orig_train_ids.json).")

    prev_used = _load_prev_used_ids(state.path(PACK))
    delta = include - prev_used
    report: Dict[str, Any] = {"split_source": split_source, "include": len(include), "prev_used": len(prev_used),
                              "delta": len(delta), "stages": {}}
    embedder = None
    if run_mode in ("all", "stage1"):
        if len(delta) == 0 and state.exists(EMB_POS) and state.exists(EMB_ALL):
            report["stages"]["stage1"] = "SKIP_NO_NEW_TRAIN_IMAGES"
        else:
            ids_to_build = include if not state.exists(EMB_POS) else (delta if len(delta) else set())
            if len(ids_to_build) > 0:
                embedder = (embedder_factory or _Embedder)(resnet50_weights, device)
                pos_tmp = state.write_path(f"{emb}/embeddings_train_pos.round_delta.npy")
                all_tmp = state.write_path(f"{emb}/embeddings_train_all.round_delta.npy")
                rep = build_embeddings_dual(coco_json=coco_json, images_dir=images_dir, save_pos_npy=pos_tmp,
                                            save_all_npy=all_tmp, include_img_ids=ids_to_build,
                                            target_cat_names={pos_name.lower()}, embedder=embedder,
                                            sample_limit=sample_limit, pad_frac=pad_frac,
                                            resolver=resolver_for("pack", images_dir, mapper))
                if rep["failed_ids"]:
                    # abort before publishing anything: the previous state stays untouched, and a
                    # retry after fixing the input processes the whole delta in reference order.
                    for tmp in (pos_tmp, all_tmp):
                        if tmp.exists():
                            tmp.unlink()
                    report["stages"]["stage1"] = "ABORTED_UNREADABLE_INPUTS"
                    report["failed_inputs"] = {str(i): rep["images"][str(i)] for i in rep["failed_ids"]}
                    report["input_resolution"] = rep["input_resolution"]
                    report["published"] = False
                    _append_receipts(state, f"{pro}/stage1_receipts.json",
                                     {"outcome": "ABORTED_UNREADABLE_INPUTS", "requested_ids": sorted(ids_to_build),
                                      "images": rep["images"], "input_resolution": rep["input_resolution"]})
                    raise PackInputError(
                        f"{len(rep['failed_ids'])} training image(s) of this round could not be read; "
                        "nothing was published and no image id was marked as used "
                        "(supply --map OLD=NEW for relocated inputs, then rerun pack)", report)
                for final, tmp in ((EMB_POS, pos_tmp), (EMB_ALL, all_tmp)):
                    if state.exists(final):
                        Z_old = np.load(state.path(final))
                        Z_new = np.load(tmp) if tmp.exists() else np.zeros((0, Z_old.shape[1]), np.float32)
                        Z_cat = Z_old if Z_new.size == 0 else (Z_new if Z_old.size == 0 else np.vstack([Z_old, Z_new]))
                        _atomic_write(state.write_path(final), lambda t, Z=Z_cat: np.save(t, Z))
                    else:
                        if tmp.exists():
                            shutil.move(str(tmp), str(state.write_path(final)))
                for tmp in (pos_tmp, all_tmp):
                    if tmp.exists():
                        tmp.unlink()
                report["stages"]["stage1"] = {"built_images": len(ids_to_build), **rep,
                                              "embed_cpu_fallbacks": embedder.fallback_count}
                _append_receipts(state, f"{pro}/stage1_receipts.json",
                                 {"outcome": "PUBLISHED", "requested_ids": sorted(ids_to_build),
                                  "contributed_ids": rep["contributed_ids"],
                                  "zero_contribution_ids": rep["zero_contribution_ids"],
                                  "pos_shape": rep["pos_shape"], "all_shape": rep["all_shape"],
                                  "images": rep["images"], "input_resolution": rep["input_resolution"]})
            else:
                report["stages"]["stage1"] = "NOTHING_TO_BUILD"

    need2 = (not state.exists(PROTO)) or (len(delta) > 0)
    if run_mode in ("all", "stage2") and need2:
        proto_tmp = state.write_path(f"{pro}/jassid_unified_proto.publishing.csv")
        out = make_unified_proto_from_embeddings(state.path(EMB_POS), proto_tmp, thresh_method=thresh_method,
                                                 fixed_thresh=fixed_thresh, percentile=percentile)
        if out[0]:
            proto_tmp.replace(state.write_path(PROTO))
        elif proto_tmp.exists():
            proto_tmp.unlink()
        report["stages"]["stage2"] = {"embed_thresh": out[1], "dim": out[2]} if out[0] else "SKIP_NO_POS"
    elif run_mode in ("all", "stage2"):
        report["stages"]["stage2"] = "SKIP_UNCHANGED"

    need3 = (not state.exists(PCA)) or (len(delta) > 0)
    if run_mode in ("all", "stage3") and need3:
        pca_tmp = state.write_path(f"{pca}/embed_pca_32.publishing.npz")
        out = fit_pca_from_embeddings(state.path(EMB_ALL), pca_tmp, k=pca_dims)
        if out[0]:
            pca_tmp.replace(state.write_path(PCA))
        elif pca_tmp.exists():
            pca_tmp.unlink()
        report["stages"]["stage3"] = {"components_shape": out[1], "dim": out[2]} if out[0] else "SKIP_TOO_FEW"
    elif run_mode in ("all", "stage3"):
        report["stages"]["stage3"] = "SKIP_UNCHANGED"

    pack: Dict[str, Any] = {"used_img_ids": sorted(set(prev_used) | set(include))}
    if state.exists(PROTO):
        df = pd.read_csv(state.path(PROTO))
        mu_cols = [c for c in df.columns if c.startswith("mu_embed_")] or [c for c in df.columns if c.startswith("mu_")]
        mu = df.iloc[0][mu_cols].to_numpy(np.float32)
        pack["mu"] = (mu / (np.linalg.norm(mu) + 1e-12)).astype(np.float32)
        pack["embed_thresh"] = float(df.iloc[0].get("embed_thresh", fixed_thresh))
    if state.exists(PCA):
        npz = np.load(state.path(PCA))
        pack["pca_components"] = np.asarray(npz["components"], np.float32)
        pack["pca_mean"] = np.asarray(npz["mean"], np.float32)
    _atomic_write(state.write_path(PACK), lambda tmp: joblib.dump(pack, tmp))
    _atomic_write(state.write_path(USED),
                  lambda tmp: tmp.write_text(json.dumps(pack["used_img_ids"], indent=2), encoding="utf-8"))
    report["published"] = True
    report["pack"] = {k: (list(v.shape) if isinstance(v, np.ndarray) else (len(v) if isinstance(v, list) else v))
                      for k, v in pack.items()}
    report["written"] = [PACK, USED] + [r for r in (EMB_POS, EMB_ALL, PROTO, PCA) if state.source_of(r) == "PROJECT_OVERLAY"]
    return report
