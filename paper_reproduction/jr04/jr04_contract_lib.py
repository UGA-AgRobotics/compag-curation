#!/usr/bin/env python3
"""Shared, model-free helpers for the JR04 sanitized-payload contract.

This module deliberately uses only the Python standard library.  It hashes files,
projects saved CSV fields, and parses CVAT XML.  It never imports, loads, or
deserializes a model and it never computes new probabilities.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


CARDS = ("IMG_9406", "IMG_9459", "IMG_9460", "IMG_9486", "IMG_9487")
EXPECTED_CARD_POINTS = {
    "IMG_9406": 228,
    "IMG_9459": 130,
    "IMG_9460": 124,
    "IMG_9486": 129,
    "IMG_9487": 129,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def file_pin(path: Path) -> dict[str, object]:
    return {"bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}


def tree_pins(root: Path) -> dict[str, dict[str, object]]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): file_pin(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def csv_projection(
    path: Path,
    columns: list[str] | tuple[str, ...] | None = None,
    *,
    delimiter: str = ",",
    aliases: dict[str, str] | None = None,
) -> dict[str, object]:
    """Hash ordered textual cell values for selected logical fields.

    `aliases` maps a logical output name to the actual input column.  This lets
    the canonical score in one table be compared with score_r92 in another
    without changing numeric text or row order.
    """
    h = hashlib.sha256()
    row_count = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        if aliases is not None:
            logical = list(aliases)
            actual = [aliases[name] for name in logical]
        else:
            actual = list(columns) if columns is not None else list(reader.fieldnames)
            logical = list(actual)
        missing = [name for name in actual if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing projected columns in {path.name}: {missing}")
        h.update(
            (json.dumps(logical, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        for row in reader:
            values = [row[name] for name in actual]
            h.update(
                (json.dumps(values, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            row_count += 1
    return {
        "columns": logical,
        "rows": row_count,
        "ordered_text_projection_sha256": h.hexdigest(),
    }


def csv_unique_key_count(path: Path, columns: tuple[str, ...]) -> int:
    values: set[tuple[str, ...]] = set()
    rows = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        missing = [name for name in columns if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing key columns in {path.name}: {missing}")
        for row in reader:
            rows += 1
            values.add(tuple(row[name] for name in columns))
    if len(values) != rows:
        raise ValueError(
            f"Duplicate scientific key in {path.name}: {rows - len(values)} duplicate rows"
        )
    return len(values)


def cvat_xml_projection(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    records: list[dict[str, object]] = []
    card_counts: dict[str, int] = {}
    h = hashlib.sha256()
    for image in root.findall("image"):
        name = image.attrib.get("name", "")
        card = Path(name).stem
        point_rows: list[dict[str, str]] = []
        count = 0
        for node in image.findall("points"):
            points = node.attrib.get("points", "")
            tokens = [token.strip() for token in points.split(";") if token.strip()]
            count += len(tokens)
            point_rows.append(
                {
                    "label": node.attrib.get("label", ""),
                    "points": points,
                    "source": node.attrib.get("source", ""),
                }
            )
        record = {
            "name": name,
            "width": image.attrib.get("width", ""),
            "height": image.attrib.get("height", ""),
            "point_shapes": point_rows,
        }
        h.update(
            (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        records.append(record)
        card_counts[card] = count
    return {
        "images": len(records),
        "points": sum(card_counts.values()),
        "points_by_card": card_counts,
        "ordered_name_dimension_label_coordinate_source_sha256": h.hexdigest(),
    }


def candidate_relative_id_snapshot(candidate: Path) -> dict[str, object]:
    metrics_path = candidate / "model_metrics__xgb_recall.csv"
    table_path = candidate / "table4_reconciled_raw_and_weighted.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        table = list(csv.DictReader(handle))
    metric_rels = [row["model_rel"] for row in metrics]
    derived_rels = [
        Path(row["model_path"]).parent.name + "/" + Path(row["model_path"]).name
        for row in table
    ]
    matches = [metric_rels.count(value) for value in derived_rels]
    return {
        "historical_model_rows": len(metrics),
        "table4_rows": len(table),
        "derived_parent_name_identifiers": derived_rels,
        "unique_model_metrics_matches_per_table4_row": matches,
        "all_table4_relative_model_ids_resolve_uniquely": matches == [1] * len(table),
    }


def scientific_contract_snapshot(
    candidate_inputs: Path, point_inputs: Path
) -> dict[str, object]:
    """Compute the model-free semantic snapshot consumed by the adapter."""
    point = Path(point_inputs)
    candidate = Path(candidate_inputs)

    candidate_scored = candidate / "scored_candidates_r92.csv"
    candidate_checkpoints = candidate / "checkpoint_scores_all_rounds.csv"
    scored_projection = csv_projection(
        candidate_scored,
        aliases={
            "img_folder": "img_folder",
            "image": "image",
            "id": "id",
            "human_label": "human_label",
            "action": "action",
            "review_weight": "review_weight",
            "r92_score": "canonical_score",
        },
    )
    checkpoint_projection = csv_projection(
        candidate_checkpoints,
        aliases={
            "img_folder": "img_folder",
            "image": "image",
            "id": "id",
            "human_label": "human_label",
            "action": "action",
            "review_weight": "review_weight",
            "r92_score": "score_r92",
        },
    )
    candidate_unique = csv_unique_key_count(candidate_scored, ("img_folder", "image", "id"))

    xml_a = cvat_xml_projection(point / "gt" / "annotations.xml")
    xml_b = cvat_xml_projection(point / "gt" / "E2E_9460_9486.xml")
    points_by_card = dict(xml_a["points_by_card"])
    overlap = set(points_by_card).intersection(xml_b["points_by_card"])
    if overlap:
        raise ValueError(f"Duplicate card across JR04 XML inputs: {sorted(overlap)}")
    points_by_card.update(xml_b["points_by_card"])

    detection_columns = [
        "image",
        "id",
        "x",
        "y",
        "w",
        "h",
        "poly",
        "xgb_p",
        "yolo_conf",
        "yolo_iou",
        "p_fused",
    ]
    tile_columns = [
        "tile_name",
        "orig_name",
        "x",
        "y",
        "crop_w",
        "crop_h",
        "orig_w",
        "orig_h",
    ]
    cards: dict[str, object] = {}
    saved_point_rows = 0
    for card in CARDS:
        base = point / "E2E" / card
        detection = csv_projection(
            base / "run_xgb_recall" / "detections.csv", detection_columns
        )
        al = csv_projection(base / "run_xgb_recall" / "al_candidates.csv")
        tiles = csv_projection(base / "tile_info" / "tiles_index.csv", tile_columns)
        saved = csv_projection(
            point
            / "E2E"
            / f"eval_{card}_xgb05_m20_reviewability"
            / "points_detail__xgb.csv"
        )
        saved_point_rows += int(saved["rows"])
        cards[card] = {
            "detections": detection,
            "al_candidates": al,
            "tile_index_scientific_fields": tiles,
            "saved_tau050_points": saved,
        }

    return {
        "candidate_population": {
            "rows": scored_projection["rows"],
            "unique_img_folder_image_id": candidate_unique,
            "scored_projection": scored_projection,
            "checkpoint_projection": checkpoint_projection,
            "same_ordered_population_labels_weights_and_r92_score": (
                scored_projection["ordered_text_projection_sha256"]
                == checkpoint_projection["ordered_text_projection_sha256"]
            ),
            "relative_model_identity_join": candidate_relative_id_snapshot(candidate),
        },
        "s3": {
            "cards": cards,
            "xml_sources": {
                "gt/annotations.xml": xml_a,
                "gt/E2E_9460_9486.xml": xml_b,
            },
            "points_by_card": points_by_card,
            "total_xml_points": int(xml_a["points"]) + int(xml_b["points"]),
            "saved_tau050_point_rows": saved_point_rows,
        },
    }


def first_difference(expected: object, actual: object, path: str = "contract") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return f"{path}: key set differs"
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: list length {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path}: {actual!r} != {expected!r}"
    return None
