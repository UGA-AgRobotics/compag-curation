"""Port of notebook cell 2 "CELL 00 - PRE-CELL MERGE".

Merges the latest human decision per ``(image, id)`` from ``review_labels.csv``
into the original-level and tiled COCO masters, using the reviewed candidate's
detection polygon (or the recorded bbox-derived rectangle fallback).  The
computation is the notebook code; the only differences are plumbing:

* inputs are read from the given paths and the updated masters are written to
  *new* paths (originals are never rewritten);
* the notebook's backup copies are replaced by the project's snapshot store.

Serialization is identical (``json.dumps(obj, indent=2)``), so the produced
JSON files are byte-identical to what the notebook would write.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

TILE_W = 512
TILE_H = 512


def load_json(p: Path, default=None):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return (default if default is not None else {})


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def parse_root(tile_name: str) -> str:
    m = re.match(r"^(?P<root>.+?)_y\d{1,8}x\d{1,8}\.(jpg|jpeg|png)$",
                 Path(tile_name).name, flags=re.IGNORECASE)
    return (m.group("root") if m else Path(tile_name).stem.split("_y")[0])


def ensure_cat(coco):
    cats = coco.get("categories", [])
    if not cats:
        coco["categories"] = [{"id": 1, "name": "CJ", "supercategory": ""},
                              {"id": 2, "name": "non-CJ", "supercategory": ""}]


def next_id(coco, key):
    cur = 0
    for it in coco.get(key, []):
        try:
            cur = max(cur, int(it.get("id", 0)))
        except Exception:
            pass
    return cur + 1


def bbox_from_poly(flat):
    xs = flat[0::2]; ys = flat[1::2]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def rect_poly_from_bbox_row(row):
    try:
        if all(k in row for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
            x = float(row["bbox_x"]); y = float(row["bbox_y"])
            w = float(row["bbox_w"]); h = float(row["bbox_h"])
            if w > 0 and h > 0:
                return [x, y, x + w, y, x + w, y + h, x, y + h]
    except Exception:
        pass
    try:
        if "bbox" in row and isinstance(row["bbox"], str) and "," in row["bbox"]:
            x, y, w, h = [float(t) for t in row["bbox"].split(",")]
            if w > 0 and h > 0:
                return [x, y, x + w, y, x + w, y + h, x, y + h]
    except Exception:
        pass
    try:
        if all(k in row for k in ("minx", "miny", "maxx", "maxy")):
            x0 = float(row["minx"]); y0 = float(row["miny"])
            x1 = float(row["maxx"]); y1 = float(row["maxy"])
            if (x1 > x0) and (y1 > y0):
                return [x0, y0, x1, y0, x1, y1, x0, y1]
    except Exception:
        pass
    return None


def resolve_detections_csv(out_root: Path, mode: str) -> Path:
    mode = (mode or "xgb_recall").strip().lower()
    if mode not in {"gate", "xgb", "xgb_recall"}:
        mode = "xgb_recall"
    det = out_root / f"run_{mode}" / "detections.csv"
    if not det.exists():
        for m in ("xgb_recall", "xgb", "gate"):
            cand = out_root / f"run_{m}" / "detections.csv"
            if cand.exists():
                det = cand
                break
    return det


def merge_reviews(
    *,
    orig_coco_in: Path,
    tiled_coco_in: Path,
    detections_csv: Path,
    reviews_csv: Path,
    orig_coco_out: Path,
    tiled_coco_out: Path,
    new_orig_ids_out: Path,
) -> dict[str, Any]:
    """Run the notebook merge; returns counters plus provenance of geometry."""

    orig = load_json(Path(orig_coco_in), {"images": [], "annotations": [], "categories": [{"id": 1, "name": "CJ"}, {"id": 2, "name": "non-CJ"}]})
    tiled = load_json(Path(tiled_coco_in), {"images": [], "annotations": [], "categories": [{"id": 1, "name": "CJ"}, {"id": 2, "name": "non-CJ"}]})
    ensure_cat(orig); ensure_cat(tiled)

    root2orig = {}
    for im in orig.get("images", []):
        fn = Path(str(im.get("file_name", ""))).name
        root = Path(fn).stem
        root2orig[root] = int(im["id"])

    det_map: dict[tuple[str, str], list[float]] = {}
    geometry_source: dict[tuple[str, str], str] = {}
    if not Path(detections_csv).exists():
        raise FileNotFoundError(f"Detections not found at {detections_csv}. Check MODE/paths.")
    with Path(detections_csv).open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            img = row.get("image")
            pid = row.get("id")
            if img is None or pid is None:
                continue
            k = (img, str(pid))
            flat = None
            src = "poly"
            poly = row.get("poly", "")
            try:
                flat = json.loads(poly) if isinstance(poly, str) else poly
            except Exception:
                flat = None
            if not (isinstance(flat, list) and len(flat) >= 6):
                flat = rect_poly_from_bbox_row(row)
                src = "bbox_rectangle_fallback"
            if isinstance(flat, list) and len(flat) >= 6:
                det_map[k] = [float(x) for x in flat]
                geometry_source[k] = src

    label_map: dict[tuple[str, str], tuple[int, int]] = {}
    roots_seen = set()
    if not Path(reviews_csv).exists():
        raise FileNotFoundError(f"Reviews CSV not found at {reviews_csv}.")
    with Path(reviews_csv).open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            img = r.get("image")
            pid = r.get("id")
            if img is None or pid is None:
                continue
            k = (img, str(pid))
            if k not in det_map:
                continue
            try:
                y = int(r.get("human_label", 0))
            except Exception:
                y = 0
            cid = 1 if y == 1 else 2
            ts = None
            try:
                ts = int(r.get("timestamp", 0))
            except Exception:
                ts = 0
            prev = label_map.get(k)
            if (prev is None) or (ts >= prev[1]):
                label_map[k] = (cid, ts)
            roots_seen.add(parse_root(img))

    new_orig_ids = []
    for root in sorted(roots_seen):
        if root not in root2orig:
            oid = next_id(orig, "images")
            orig["images"].append({"id": int(oid), "file_name": f"{root}.jpg"})
            root2orig[root] = int(oid)
            new_orig_ids.append(int(oid))

    name2tid = {Path(im["file_name"]).name: int(im["id"]) for im in tiled.get("images", [])}
    for (img, _) in label_map.keys():
        bn = Path(img).name
        if bn not in name2tid:
            tid = next_id(tiled, "images")
            root = parse_root(bn)
            tiled["images"].append({
                "id": int(tid),
                "width": TILE_W, "height": TILE_H,
                "file_name": bn,
                "meta": {
                    "orig_file_name": f"{root}.jpg",
                    "orig_image_id": int(root2orig[root]),
                },
            })
            name2tid[bn] = int(tid)

    revkey2annidx = {}
    for idx, a in enumerate(tiled.get("annotations", [])):
        meta = a.get("meta", {})
        rid = meta.get("review_id", None)
        if rid is None:
            continue
        revkey2annidx[(int(a.get("image_id", -1)), int(rid))] = idx

    added, updated, skipped = 0, 0, 0
    ann_id = next_id(tiled, "annotations")
    fallback_rect = 0
    for (img, pid), (cid, _) in label_map.items():
        bn = Path(img).name
        image_id = name2tid[bn]
        flat = det_map[(img, pid)]
        if geometry_source[(img, pid)] != "poly":
            fallback_rect += 1
        seg = [[float(x) for x in flat]]
        bbox = bbox_from_poly(flat)
        rid = int(pid)
        key = (int(image_id), rid)
        if key in revkey2annidx:
            a = tiled["annotations"][revkey2annidx[key]]
            a["category_id"] = int(cid)
            a["segmentation"] = seg
            a["bbox"] = bbox
            a["area"] = float(bbox[2] * bbox[3])
            meta = a.get("meta", {}) or {}
            meta["review_id"] = rid
            a["meta"] = meta
            updated += 1
        else:
            tiled["annotations"].append({
                "id": int(ann_id),
                "image_id": int(image_id),
                "category_id": int(cid),
                "segmentation": seg,
                "bbox": bbox,
                "iscrowd": 0,
                "area": float(bbox[2] * bbox[3]),
                "meta": {"review_id": rid},
            })
            revkey2annidx[key] = len(tiled["annotations"]) - 1
            ann_id += 1
            added += 1

    save_json(Path(orig_coco_out), orig)
    save_json(Path(tiled_coco_out), tiled)
    save_json(Path(new_orig_ids_out), sorted(list(map(int, new_orig_ids))))
    return {
        "reviews_matched": len(label_map), "added": added, "updated": updated, "skipped": skipped,
        "tiled_images": len(tiled.get("images", [])), "tiled_annotations": len(tiled.get("annotations", [])),
        "new_orig_ids": sorted(new_orig_ids),
        "geometry": {"poly": len(label_map) - fallback_rect, "bbox_rectangle_fallback": fallback_rect,
                     "note": "bbox-derived rectangles are recorded as such, not as human boundary tracing"},
    }
