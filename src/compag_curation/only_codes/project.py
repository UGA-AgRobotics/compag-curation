"""Import an existing Only_codes workspace as an ``only-codes-compat-v1`` project.

The import never rewrites the original files.  It records:

* the original roots (legacy ``Clean`` root, SAM2 repository, torch hub) and
  explicit absolute-path prefix mappings;
* the legacy relative layout taken from the notebook bootstrap literals;
* a dependency inventory per entry point with SHA-256 identities;
* trusted classifier conversions into pickle-free bundles;
* the persisted compatibility contract (profile + contract hash).

Reopening a project whose persisted profile/contract hash differs from the
running code is refused, so an imported project never silently switches to a
modern algorithm.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import PROFILE_ID
from .contract import contract, contract_sha256
from .state import LegacyState, PathMapper, sha256_file

PROJECT_SCHEMA = "compag-only-codes-project/v1"
PROJECT_MANIFEST = "ONLY_CODES_PROJECT.json"
TRANSITIONS = "TRANSITIONS.jsonl"

# Bootstrap literals of notebook cell 1, relative to the legacy Clean root.
LAYOUT: dict[str, str] = {
    "pc_codes": "Shared/PC_codes",
    "orig_coco": "Shared/PC_codes/annotations/CJ_NOCJ_e.json",
    "tiled_coco": "Shared/PC_codes/annotations/CJ_NOCJ_tiles_512.json",
    "tiles_dir": "Shared/PC_codes/img_tiles_512",
    "art_root": "Shared/PC_codes/npy_npz_csv/tiles_512",
    "split_dir": "Shared/PC_codes/npy_npz_csv/tiles_512/splits",
    "feat_out_dir": "Shared/PC_codes/annotation_2_features/cj_noncj_512_tiles",
    "features_csv": "Shared/PC_codes/annotation_2_features/cj_noncj_512_tiles/features_train.csv",
    "maskout_tile_root": "Shared/maskout_tile",
    "model_root": "always_same/jupyter/models/cj_classifier",
    "infer_proto": "always_same/stage2/train_foldA/jassid_unified_proto.csv",
    "infer_proto_padded": "always_same/stage2/train_foldA/jassid_unified_proto_padded.csv",
    "infer_pca": "always_same/stage3/train_foldA/embed_pca_32.npz",
    "yolo_weights": "Shared/PC_codes/datasets/cj_yolo_full/runs/yolo11n_full/weights/best.pt",
}
FOLD_TAG = "train_foldA"
POS_NAME = "cj"
BOOTSTRAP_MODEL_NAME = "cj_ultra_tilesafe_xgb_r17_hybrid.pkl"
SAM2_CKPT_REL = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "sam2.1/sam2.1_hiera_l"
RESNET50_REL = "hub/checkpoints/resnet50-11ad3fa6.pth"


class ProjectError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tree_identity(root: Path, patterns: Iterable[str] = ("*",)) -> dict[str, Any]:
    files = []
    for pat in patterns:
        files.extend(p for p in root.glob(pat) if p.is_file())
    files = sorted(set(files), key=lambda p: p.name)
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return {"count": len(files), "names_in_enumeration_order": [p.name for p in files][:5] + (["..."] if len(files) > 5 else []),
            "tree_sha256": h.hexdigest()}


def discover_rounds(model_root: Path) -> list[dict[str, Any]]:
    out = []
    if not model_root.exists():
        return out
    for d in sorted(model_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        m = re.match(r"^r(\d+)(?:_(.*))?$", d.name)
        if not m:
            continue
        pkls = sorted(p.name for p in d.glob("*.pkl"))
        out.append({
            "dir": d.name, "round": int(m.group(1)),
            "models": pkls,
            "has_tile_preds": (d / "tile_preds.csv").exists(),
            "thresholds": sorted(p.name for p in d.glob("threshold_*.json")),
            "has_best_params": (d / "best_params_used.json").exists(),
        })
    out.sort(key=lambda r: r["round"])
    return out


def discover_images(maskout_root: Path) -> list[dict[str, Any]]:
    out = []
    if not maskout_root.exists():
        return out
    for d in sorted(maskout_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        m = re.match(r"^IMG_(\d+)$", d.name)
        if not m:
            continue
        runs = sorted(r.name for r in d.glob("run_*") if (r / "detections.csv").exists())
        out.append({
            "name": d.name, "number": int(m.group(1)),
            "has_tiles": (d / "tiles").is_dir(),
            "runs_with_detections": runs,
            "review_labels": (d / "review_labels.csv").exists(),
            "review_label_variants": sorted(p.name for p in d.glob("review_labels*.csv")),
            "round_k_new_orig_ids": (d / "round_k_new_orig_ids.json").exists(),
            "round_k_new_orig_ids_applied": (d / "round_k_new_orig_ids.applied.json").exists(),
            "backup_dir_marker": (d / ".backup_dir").exists(),
        })
    return out


# ---------------------------------------------------------------- inventory
_ENTRY_POINTS = {
    "predict": ["tiles_of_image", "model_bundle", "infer_proto_padded|infer_proto", "infer_pca", "sam2_ckpt",
                "sam2_config", "resnet50", "yolo_weights"],
    "review": ["detections_of_image", "tiles_of_image"],
    "merge": ["orig_coco", "tiled_coco", "detections_of_image", "review_labels_of_image"],
    "splits": ["orig_coco", "split_dir"],
    "pack": ["tiled_coco", "tiles_dir", "split_dir", "art_root", "resnet50"],
    "features": ["tiled_coco", "tiles_dir", "training_pack", "used_img_ids", "resnet50"],
    "train": ["features_csv", "model_root"],
}


def _row(input_type, source_reader, new_reader, modes, path, sha, transform, status, note=""):
    return {"input_type": input_type, "old_source_reader": source_reader, "new_importer_reader": new_reader,
            "required_modes": modes, "resolved_path": path, "sha256_or_tree": sha,
            "transformation_performed": transform, "verification_status": status, "note": note}


def build_inventory(legacy_root: Path, sam2_repo: Path | None, torch_home: Path | None,
                    *, image: str | None, hash_large: bool = True) -> list[dict[str, Any]]:
    L = {k: legacy_root / v for k, v in LAYOUT.items()}
    rows: list[dict[str, Any]] = []

    def file_row(key, input_type, reader, new_reader, modes, transform="none (read in place)", big=False):
        p = L[key]
        if p.is_file():
            sha = sha256_file(p) if (hash_large or not big) else f"size={p.stat().st_size}"
            rows.append(_row(input_type, reader, new_reader, modes, str(p), sha, transform, "FOUND_HASHED"))
        else:
            rows.append(_row(input_type, reader, new_reader, modes, str(p), "", transform, "MISSING"))

    file_row("orig_coco", "original-level COCO JSON", "cell 2/3 json.loads", "only_codes.merge / splits", "merge,splits",
             "merge writes an updated copy into the project overlay (same relative path)")
    file_row("tiled_coco", "tiled COCO JSON (polygons/RLE, ids, categories, meta.review_id)", "cells 2/4/5 json.loads",
             "only_codes.merge / pack / features", "merge,pack,features",
             "merge writes an updated copy into the project overlay", big=True)
    tiles = L["tiles_dir"]
    if tiles.is_dir():
        rows.append(_row("training tile images", "cv2.imread(IMAGES_DIR/file_name)", "same reader, original files",
                         "pack,features", str(tiles), _tree_identity(tiles, ("*.jpg", "*.png", "*.jpeg"))["tree_sha256"],
                         "none (original bytes, names and dimensions)", "FOUND_TREE_HASHED"))
    else:
        rows.append(_row("training tile images", "cv2.imread", "same", "pack,features", str(tiles), "", "none", "MISSING"))
    split = L["split_dir"]
    for name in ("orig_train_ids.json", "orig_val_ids.json", "orig_test_ids.json", "used_img_ids_train.json"):
        p = split / name
        rows.append(_row(f"split file {name}", "cells 3/4 json.loads", "only_codes.splits / pack", "splits,pack",
                         str(p), sha256_file(p) if p.exists() else "", "splits updated in project overlay",
                         "FOUND_HASHED" if p.exists() else ("OPTIONAL_ABSENT" if name == "used_img_ids_train.json" else "MISSING")))
    art = L["art_root"]
    for rel, kind, big in (
        (f"stage1_embeddings/{FOLD_TAG}/embeddings_train_pos.npy", "POS embeddings (Stage-1)", True),
        (f"stage1_embeddings/{FOLD_TAG}/embeddings_train_all.npy", "ALL embeddings (Stage-1)", True),
        (f"stage2_proto/{FOLD_TAG}/jassid_unified_proto.csv", "training prototype CSV", False),
        (f"stage2_proto/{FOLD_TAG}/foldsafe_pack.joblib", "training fold-safe pack (joblib data)", False),
        (f"stage2_proto/{FOLD_TAG}/used_img_ids.json", "pack used image ids", False),
        (f"stage3_pca/{FOLD_TAG}/embed_pca_32.npz", "training PCA", False),
    ):
        p = art / rel
        reader = "restricted NumPy-only joblib reader (no arbitrary pickle)" if rel.endswith(".joblib") else "numpy/pandas/json"
        rows.append(_row(kind, "cells 4/5", reader, "pack,features", str(p),
                         (sha256_file(p) if (hash_large or not big) else f"size={p.stat().st_size}") if p.exists() else "",
                         "incremental updates written to project overlay", "FOUND_HASHED" if p.exists() else "MISSING"))
    file_row("features_csv", "training feature table (features_train.csv)", "cell 3A read_features_csv_robust; cell 2 resume keys",
             "only_codes.training / features", "features,train",
             "none; new rows appended to a project copy only when needed; provenance sidecar", big=True)
    for key, kind in (("infer_proto_padded", "inference prototype (padded) CSV"), ("infer_proto", "inference prototype CSV"),
                      ("infer_pca", "inference PCA (always_same, separate from training PCA)"),
                      ("yolo_weights", "YOLO11n weights (hybrid detector)")):
        file_row(key, kind, "runner/sam2_and_filter", "only_codes.inference", "predict",
                 "prototype re-padded through pandas into the run work dir (LB-13)" if key.startswith("infer_proto") else "none")
    mr = L["model_root"]
    rounds = discover_rounds(mr)
    rows.append(_row("round model directory tree", "cells 3A/3B prev-round scan; runner CJ_MODEL_ROOT", "only_codes.training / project",
                     "train,predict", str(mr), f"rounds={len(rounds)}", "read-only; new rounds written to overlay",
                     "FOUND" if rounds else "MISSING"))
    hist = mr / "history.csv"
    rows.append(_row("training history.csv", "cell 3B append", "only_codes.training (append to overlay copy)", "train", str(hist),
                     sha256_file(hist) if hist.exists() else "", "append in overlay copy", "FOUND_HASHED" if hist.exists() else "OPTIONAL_ABSENT"))
    if sam2_repo is not None:
        ck = sam2_repo / SAM2_CKPT_REL
        rows.append(_row("SAM2.1 Hiera-L checkpoint", "build_sam2(config, ckpt)", "same", "predict", str(ck),
                         sha256_file(ck) if ck.exists() else "", "none", "FOUND_HASHED" if ck.exists() else "MISSING"))
    if torch_home is not None:
        rw = torch_home / RESNET50_REL
        rows.append(_row("ResNet50 IMAGENET1K_V2 weights", "torchvision resnet50(weights=IMAGENET1K_V2) via TORCH_HOME",
                         "explicit local state dict (same file)", "predict,pack,features", str(rw),
                         sha256_file(rw) if rw.exists() else "", "none", "FOUND_HASHED" if rw.exists() else "MISSING"))
    if image:
        d = legacy_root / LAYOUT["maskout_tile_root"] / image
        t = d / "tiles"
        rows.append(_row(f"inference tiles of {image}", "pipeline img_dir.glob('*.*') sorted", "same enumeration",
                         "predict,review", str(t), _tree_identity(t, ("*.*",))["tree_sha256"] if t.is_dir() else "",
                         "none (original bytes)", "FOUND_TREE_HASHED" if t.is_dir() else "MISSING"))
        for rel, kind in (("run_xgb_recall/detections.csv", "detections.csv (candidate geometry/scores)"),
                          ("run_xgb_recall/al_candidates.csv", "AL shortlist"),
                          ("run_xgb_recall/features_pool.csv", "inference feature export"),
                          ("run_xgb_recall/gate_summary.csv", "gate summary"),
                          ("run_xgb_recall/_dash_cache/seen.json", "review GUI seen state"),
                          ("review_labels.csv", "review log (append-only)"),
                          ("round_k_new_orig_ids.json", "merge output new original ids"),
                          ("round_k_new_orig_ids.applied.json", "consumed merge ids")):
            p = d / rel
            rows.append(_row(f"{image}: {kind}", "GUI / merge / feature export", "only_codes.review / merge / features",
                             "review,merge,features", str(p), sha256_file(p) if p.exists() else "",
                             "copied into the project overlay before any append", "FOUND_HASHED" if p.exists() else "ABSENT"))
        for p in sorted(d.glob("review_labels*.csv")):
            if p.name != "review_labels.csv":
                rows.append(_row(f"{image}: review log variant {p.name}", "not read by OLD (name differs)",
                                 "only by explicit --review-labels selection", "review,merge", str(p), sha256_file(p),
                                 "none; never substituted automatically", "FOUND_NOT_AUTO_SELECTED"))
    return rows


def write_compatibility_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["input_type", "old_source_reader", "new_importer_reader", "required_modes", "resolved_path",
              "sha256_or_tree", "transformation_performed", "verification_status", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------- project object
@dataclass
class Project:
    root: Path
    manifest: dict[str, Any]

    @property
    def legacy_root(self) -> Path:
        return Path(self.manifest["roots"]["legacy_root"])

    @property
    def state(self) -> LegacyState:
        return LegacyState(self.legacy_root, self.root / "state")

    @property
    def mapper(self) -> PathMapper:
        return PathMapper([(r["old_prefix"], r["new_prefix"]) for r in self.manifest.get("path_mappings", [])])

    def sam2_ckpt(self) -> Path:
        return Path(self.manifest["roots"]["sam2_repo_root"]) / SAM2_CKPT_REL

    def resnet50(self) -> Path:
        return Path(self.manifest["roots"]["torch_home"]) / RESNET50_REL

    def bundle_for(self, round_dir: str) -> Path:
        b = self.manifest.get("model_bundles", {}).get(round_dir)
        if not b:
            raise ProjectError(f"no verified model bundle for {round_dir}; run 'only-codes import-model' first")
        return self.root / b["bundle_dir"]

    def record_transition(self, action: str, detail: dict[str, Any]) -> None:
        entry = {"time_utc": _now(), "action": action, "profile": PROFILE_ID, "contract_sha256": contract_sha256(), **detail}
        with (self.root / TRANSITIONS).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")

    def save(self) -> None:
        tmp = self.root / (PROJECT_MANIFEST + ".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.root / PROJECT_MANIFEST)


def open_project(root: Path | str) -> Project:
    root = Path(root)
    mp = root / PROJECT_MANIFEST
    if not mp.exists():
        raise ProjectError(f"not an only-codes-compat project: {root}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if manifest.get("schema") != PROJECT_SCHEMA or manifest.get("profile") != PROFILE_ID:
        raise ProjectError("project profile is not only-codes-compat-v1; refusing to switch algorithms")
    if manifest.get("contract_sha256") != contract_sha256():
        raise ProjectError("persisted compatibility contract differs from this application build; "
                           "refusing to reopen with a different algorithm (re-import explicitly to change)")
    return Project(root, manifest)


def import_project(*, legacy_root: Path, output: Path, sam2_repo_root: Path, torch_home: Path,
                   image: str | None = None, path_mappings: list[tuple[str, str]] | None = None,
                   hash_large: bool = True) -> Project:
    legacy_root = Path(legacy_root).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise ProjectError(f"project output already exists (no-clobber): {output}")
    if output.resolve().is_relative_to(legacy_root):
        raise ProjectError("the project must be outside the original workspace (originals are never modified)")
    rows = build_inventory(legacy_root, Path(sam2_repo_root), Path(torch_home), image=image, hash_large=hash_large)
    output.mkdir(parents=True)
    (output / "state").mkdir()
    (output / "bundles").mkdir()
    manifest = {
        "schema": PROJECT_SCHEMA,
        "profile": PROFILE_ID,
        "contract_sha256": contract_sha256(),
        "contract": contract(),
        "created_utc": _now(),
        "roots": {"legacy_root": str(legacy_root), "sam2_repo_root": str(Path(sam2_repo_root)),
                  "torch_home": str(Path(torch_home))},
        "path_mappings": PathMapper(list(path_mappings or [])).to_dict(),
        "layout": LAYOUT,
        "fold_tag": FOLD_TAG,
        "pos_name": POS_NAME,
        "bootstrap_model_name": BOOTSTRAP_MODEL_NAME,
        "sam2_config": SAM2_CONFIG,
        "current_image": image,
        "rounds_discovered": discover_rounds(legacy_root / LAYOUT["model_root"]),
        "images_discovered": discover_images(legacy_root / LAYOUT["maskout_tile_root"]),
        "inventory": rows,
        "model_bundles": {},
        "trusted_pickles": [],
        "originals_policy": "READ_ONLY: every update is written to <project>/state with the legacy relative layout",
    }
    proj = Project(output, manifest)
    write_compatibility_matrix(rows, output / "INPUT_COMPATIBILITY_MATRIX.csv")
    proj.save()
    proj.record_transition("import", {"legacy_root": str(legacy_root), "image": image,
                                      "inventory_rows": len(rows),
                                      "missing": [r["input_type"] for r in rows if r["verification_status"] == "MISSING"]})
    return proj


def import_model(proj: Project, round_dir: str, *, trusted_sha256: str, model_file: str | None = None,
                 verification_rows_max: int = 3000) -> dict[str, Any]:
    """Convert one trusted round model into a verified pickle-free bundle."""

    import pandas as pd

    from .model_bundle import convert_trusted_legacy_pickle

    rel_dir = f"{LAYOUT['model_root']}/{round_dir}"
    d = proj.state.path(rel_dir)
    pkls = sorted(d.glob("*.pkl"))
    if model_file:
        pkls = [d / model_file]
    if len(pkls) != 1 or not pkls[0].exists():
        raise ProjectError(f"expected exactly one model .pkl in {d}, found {[p.name for p in pkls]}")
    rows = None
    fcsv = proj.state.path(LAYOUT["features_csv"])
    if fcsv.exists():
        rows = pd.read_csv(fcsv, nrows=verification_rows_max, low_memory=False)
    out_rel = f"bundles/{round_dir}"
    manifest = convert_trusted_legacy_pickle(pkls[0], proj.root / out_rel, trusted_sha256=trusted_sha256,
                                             verification_rows=rows)
    proj.manifest["model_bundles"][round_dir] = {"bundle_dir": out_rel, "source": manifest["source"],
                                                  "verification": manifest["verification"]}
    proj.manifest["trusted_pickles"].append({"path": str(pkls[0]), "sha256": trusted_sha256, "time_utc": _now()})
    proj.save()
    proj.record_transition("import-model", {"round_dir": round_dir, "sha256": trusted_sha256,
                                            "verification": manifest["verification"]["status"]})
    return manifest


def next_image_name(maskout_root_views: list[Path], cur: str) -> str:
    """Runner / GUI-patch rule: smallest IMG_n with n > current that is a directory."""

    pat = re.compile(r"^IMG_(\d+)$")
    m = pat.match(cur)
    if not m:
        return cur
    cur_n = int(m.group(1))
    cands = set()
    for root in maskout_root_views:
        if not root.exists():
            continue
        for name in os.listdir(root):
            mm = pat.match(name)
            if mm and int(mm.group(1)) > cur_n and (root / name).is_dir():
                cands.add((int(mm.group(1)), name))
    return sorted(cands)[0][1] if cands else cur


def alloc_next_model_name(model_root_views: list[Path], model_name: str = BOOTSTRAP_MODEL_NAME) -> tuple[str, str]:
    """Port of the bootstrap ``_alloc_next_model_name`` over original + overlay."""

    m = re.search(r"(?P<prefix>.*_xgb_)r(?P<num>\d+)(?P<suffix>_hybrid\.pkl)$", model_name)
    if not m:
        raise ProjectError(f"cannot parse round from model name {model_name}")
    prefix, num, suffix = m.group("prefix"), int(m.group("num")), m.group("suffix")
    tag = f"r{num}_hybrid"
    while any((root / tag).exists() for root in model_root_views):
        num += 1
        tag = f"r{num}_hybrid"
    return f"{prefix}r{num}{suffix}", tag
