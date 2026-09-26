"""Synthetic counterexamples for the r92 review bridge; no human labels or photo claims."""

import csv
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from compag_curation.r92_sample import load_model, score_csv
from compag_curation.r92_review import make_session


model_path = Path(os.environ["COMPAG_TEST_R92_MODEL"])
model = load_model(model_path)
outcomes = {}
with tempfile.TemporaryDirectory(prefix="r92 bridge guards ") as temp:
    root = Path(temp)
    for case in ("colliding_review_keys", "missing_geometry", "missing_tile", "valid_geometry_non512"):
        sub = root / case
        sub.mkdir()
        tiles = sub / "tiles"
        tiles.mkdir()
        tile = "SYNTHETIC_y00000x00000.jpg"
        if case != "missing_tile":
            size = 64 if case == "valid_geometry_non512" else 512
            assert cv2.imwrite(str(tiles / tile), np.full((size, size, 3), 210, np.uint8))
        table = sub / "features.csv"
        with table.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(["img_folder","image","id","bbox_x","bbox_y","bbox_w","bbox_h","poly",*model.order])))
            writer.writeheader()
            for index in range(2 if case == "colliding_review_keys" else 1):
                values = {name: repr(float(model.medians[j])) for j, name in enumerate(model.order)}
                values.update({"img_folder":f"FOLDER_{index}", "image":tile, "id":"1",
                               "bbox_x":"" if case == "missing_geometry" else "10",
                               "bbox_y":"" if case == "missing_geometry" else "10",
                               "bbox_w":"" if case == "missing_geometry" else "20",
                               "bbox_h":"" if case == "missing_geometry" else "20", "poly":""})
                writer.writerow(values)
        scores = sub / "scores"
        score_csv(model_path, table, scores)
        state = sub / "state"
        try:
            make_session(scores / "detections.csv", tiles, state)
        except ValueError as exc:
            outcomes[case] = str(exc)
        else:
            raise AssertionError(f"r92 bridge accepted {case}")
        assert not state.exists(), f"failed guard left review state for {case}"
assert len(outcomes) == 4
print(json.dumps({"status":"PASS","scope":"SYNTHETIC_ADAPTER_PRECONDITION","outcomes":outcomes}))
