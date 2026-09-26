"""Synthetic r92 prediction -> existing HTTP reviewer -> guarded COCO export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from compag_curation.r92_sample import load_model, score_csv
from compag_curation.r92_review import export, make_session
from compag_curation.only_codes.review_web import create_server

model_path = Path(os.environ["COMPAG_TEST_R92_MODEL"])
model = load_model(model_path)
with tempfile.TemporaryDirectory(prefix="r92 review with spaces ") as temp:
    root = Path(temp)
    table = root / "synthetic.csv"
    tile_name = "SYNTHETIC_y00000x00000.jpg"
    tiles = root / "tiles"
    tiles.mkdir()
    assert cv2.imwrite(str(tiles / tile_name), np.full((512, 512, 3), 210, np.uint8))
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["img_folder", "image", "id", *model.order, "poly"])
        writer.writeheader()
        for i in (1, 2):
            values = {name: repr(float(model.medians[j])) for j, name in enumerate(model.order)}
            for name in model.order:
                if name.startswith("g_"):
                    values[name] = ""
            values.update(bbox_x=str(i * 15), bbox_y="4", bbox_w="8", bbox_h="7")
            writer.writerow({"img_folder":"SYNTHETIC", "image":tile_name,"id":str(i),
                             **values,"poly":"[15,4,23,4,23,11,15,11]" if i == 1 else ""})
    source_hash = hashlib.sha256(table.read_bytes()).hexdigest()
    score = root / "scores"
    score_receipt = score_csv(model_path, table, score)
    assert score_receipt["row_count"] == 2
    state_dir = root / "state"
    session = make_session(score / "detections.csv", tiles, state_dir)
    assert len(session.items("SYNTHETIC")) == 2
    server, url = create_server(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["token"][0]
    root_url = f"http://127.0.0.1:{server.server_address[1]}"

    def request(path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"X-Review-Token": token}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(root_url + path, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            return json.loads(content) if path.startswith("/api/") else content

    try:
        assert b"COMPAG r92" in request("/?token=" + urllib.parse.quote(token))
        view = request("/api/state")
        items = {item["key"]: item for item in view["items"]}
        assert items[tile_name + "||1"]["geometry"] == "poly"
        assert items[tile_name + "||2"]["geometry"] == "bbox_rectangle"
        categories = [{"id":1,"name":"CJ"},{"id":2,"name":"non-CJ"}]
        (root / "originals").mkdir()
        assert cv2.imwrite(str(root / "originals/SYNTHETIC.jpg"), np.full((512, 512, 3), 210, np.uint8))
        orig_coco = {"images":[{"id":1,"file_name":"SYNTHETIC.jpg","width":512,"height":512}],
                     "annotations":[],"categories":categories}
        tiled_coco = {"images":[{"id":1,"file_name":tile_name,"width":512,"height":512,
                                 "meta":{"orig_file_name":"SYNTHETIC.jpg","orig_image_id":1,"offset_x":0,"offset_y":0,
                                         "warped_width":512,"warped_height":512,"warp_mode":"identity_fallback",
                                         "inverse_matrix":[[1,0,0],[0,1,0],[0,0,1]]}}],
                      "annotations":[],"categories":categories}
        orig = root / "orig.json"
        tiled = root / "tiled.json"
        orig.write_text(json.dumps(orig_coco))
        tiled.write_text(json.dumps(tiled_coco))
        try:
            export(state_dir, orig, tiled, root / "incomplete", confirm_review_complete=True)
        except ValueError as exc:
            assert "incomplete" in str(exc)
        else:
            raise AssertionError("incomplete review was accepted")
        for action, key in [("accept", tile_name + "||1"), ("flip", tile_name + "||2"),
                            ("accept", tile_name + "||2")]:
            result = request("/api/action", {"action":action,"key":key,"base":view["base"]})
            assert "error" not in result, result
        assert request("/api/state")["n_labels"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    reopened = make_session(score / "detections.csv", tiles, state_dir)
    assert reopened.labels_eff[tile_name + "||2"]["action"] == "accept"
    with (state_dir / "review_labels.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    orig_before = orig.read_bytes()
    tiled_before = tiled.read_bytes()
    report = export(state_dir, orig, tiled, root / "export", confirm_review_complete=True)
    result = json.loads((root / "export/tiled_coco.json").read_text())
    assert len(result["annotations"]) == 2
    assert result["images"][0]["width"] == result["images"][0]["height"] == 512
    assert {a["category_id"] for a in result["annotations"]} <= {1, 2}
    assert len({a["id"] for a in result["annotations"]}) == 2
    assert orig_before == orig.read_bytes() and tiled_before == tiled.read_bytes()
    assert source_hash == hashlib.sha256(table.read_bytes()).hexdigest()
    auto_state = root / "auto_state"
    auto = make_session(score / "detections.csv", tiles, auto_state)
    auto.act("auto_accept_rest", base="SYNTHETIC")
    try:
        export(auto_state, orig, tiled, root / "auto_export", confirm_review_complete=True)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("unreviewed bulk auto-accept was treated as human completion")
    assert not (root / "auto_export").exists()
    print(json.dumps({"status":"PASS","fixture":"SYNTHETIC_HTTP_API_NOT_HUMAN",
                      "score_rows":2,"review_actions":3,"effective_labels":2,
                      "exported_annotations":len(result["annotations"]),
                      "source_inputs_unchanged":True,"report":report},default=str))
