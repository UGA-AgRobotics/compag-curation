"""Hash-pinned, pickle-free r92 CPU table scoring and local sample access.

This is a separate CPU tabular adapter. The historical Stage-20 CUDA transfer
contract in ``review.published_model`` is deliberately unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ARCHIVE_NAME = "COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip"
ARCHIVE_BYTES = 3704653
ARCHIVE_SHA256 = "fffef7a85192b99dcca3bcc5750215150d2ca9659121363198041244bbfe57c1"
MANIFEST_SHA256 = "18b1dd35572163357e698d0a92b40ea31cf5f95ea88c0fc11715d883a14cf78f"
CLASSIFIER_SHA256 = "ebcd0761bd4d8b5db466e2724c6c15882286fdc28ab29633c9f4c49350375bc4"
FEATURE_ORDER_SHA256 = "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"
PAPER_THRESHOLD = 0.5
LEGACY_THRESHOLD = 0.5843676924705505
ROOT_NAME = "compag_cj_r92"
ASSETS = {
    "classifier.ubj": CLASSIFIER_SHA256,
    "imputer.json": "7f3e53a571bd8d38db7e780806d3dd6a773282e11c5d9874ff5d94f5e5487361",
    "feature_schema.json": "3af194c021b6138a0139bca7b1c456b024004e8ed8845412de1060f74115f128",
    "feature_state.json": "b0632fc32850ae0f0823dbee73a1a20b0187ce4cc9c1e4b992ca40e2840a02df",
    "pca32_components.npy": "a77783f1a647bea715d1c633083a18782ee28f99ecb8f3482130caa1980df897",
    "pca32_mean.npy": "3ec604b75220370a4ae66445e9b411cb66861d3c0da8bcae64d55abdbb2bba31",
    "prototype.npy": "51592613e7e048bd75e998e6ac127098c5df7b99b8dcbfd5852c24549de6a1f8",
    "threshold.json": "ab0208834625345158967afa6d7958cfc7b64fe37768254e6bda92f71145db33",
    "class_map.json": "8dfef6137128b00d601fcb8e03471d4bd4647ab73980c8fb6fb9c9ff1c8e5f1c",
    "MODEL_CARD.md": "a3d12f40858d1ac59eeeb88d9bd2d9236fcb6f5c239a84d49b5fe7c5fd3dc268",
    "NOTICE.md": "d5c3f5f08e07e06cb580ee1273d97ac0fe8477516c39fa937da8a5b655a40de0",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(data: bytes, role: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid {role} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {role} object")
    return value


def _npy(data: bytes, shape: tuple[int, ...], role: str) -> None:
    import numpy as np

    if not data.startswith(b"\x93NUMPY"):
        raise ValueError(f"{role} is not NPY")
    stream = io.BytesIO(data)
    array = np.load(stream, allow_pickle=False)
    if stream.tell() != len(data) or array.shape != shape or array.dtype != np.dtype("<f4"):
        raise ValueError(f"{role} shape/dtype/trailing bytes changed")
    if not np.isfinite(array).all():
        raise ValueError(f"{role} contains nonfinite values")


def _payloads(path: Path) -> dict[str, bytes]:
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_file():
        if path.stat().st_size != ARCHIVE_BYTES or _file_sha(path) != ARCHIVE_SHA256:
            raise ValueError("r92 archive size/hash differs from the pinned sample")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            expected = {f"{ROOT_NAME}/{name}" for name in (*ASSETS, "MANIFEST.json")}
            if len(names) != len(expected) or set(names) != expected or archive.testzip() is not None:
                raise ValueError("r92 archive members or CRC changed")
            data = {name: archive.read(f"{ROOT_NAME}/{name}") for name in (*ASSETS, "MANIFEST.json")}
    elif path.is_dir():
        if path.name != ROOT_NAME or path.is_symlink():
            raise ValueError("r92 extracted directory must be named compag_cj_r92")
        expected = set(ASSETS) | {"MANIFEST.json"}
        if {p.name for p in path.iterdir()} != expected:
            raise ValueError("r92 extracted directory members changed")
        data = {}
        for name in expected:
            member = path / name
            if member.is_symlink() or not member.is_file() or member.stat().st_nlink != 1:
                raise ValueError(f"unsafe r92 member: {name}")
            data[name] = member.read_bytes()
    else:
        raise ValueError("r92 model path must be its ZIP or extracted directory")
    if _sha(data["MANIFEST.json"]) != MANIFEST_SHA256:
        raise ValueError("r92 sample manifest hash changed")
    manifest = _json(data["MANIFEST.json"], "manifest")
    if manifest.get("schema") != "compag-cj-r92-inference-sample/v1" or manifest.get("sample_version") != "1.0.0rc1":
        raise ValueError("r92 sample identity changed")
    rows = manifest.get("members")
    if not isinstance(rows, list) or {r.get("path") for r in rows if isinstance(r, dict)} != set(ASSETS) or len(rows) != len(ASSETS):
        raise ValueError("r92 manifest member list changed")
    for row in rows:
        name = row["path"]
        if set(row) != {"path", "bytes", "sha256"} or row["sha256"] != ASSETS[name] or row["bytes"] != len(data[name]) or _sha(data[name]) != ASSETS[name]:
            raise ValueError(f"r92 member identity changed: {name}")
    feature = _json(data["feature_schema.json"], "feature schema")
    imputer = _json(data["imputer.json"], "imputer")
    classes = _json(data["class_map.json"], "class map")
    threshold = _json(data["threshold.json"], "threshold")
    state = _json(data["feature_state.json"], "feature state")
    from .canonical.spec import CANONICAL_FEATURE_ORDER

    order = list(CANONICAL_FEATURE_ORDER)
    if len(order) != 93 or feature.get("feature_order") != order or imputer.get("feature_order") != order:
        raise ValueError("r92 feature order changed")
    if feature.get("feature_order_sha256") != FEATURE_ORDER_SHA256 or imputer.get("feature_order_sha256") != FEATURE_ORDER_SHA256:
        raise ValueError("r92 feature-order hash changed")
    if imputer.get("strategy") != "median_train_only" or set(imputer.get("statistics", {})) != set(order):
        raise ValueError("r92 imputer policy changed")
    if classes != {"schema":"compag-curation-r92-class-map/v1","positive_class":1,"classes":[{"classifier_value":0,"display_label":"Non-CJ"},{"classifier_value":1,"display_label":"CJ"}]}:
        raise ValueError("r92 class map changed")
    if threshold.get("published_fixed_threshold") != PAPER_THRESHOLD or threshold.get("legacy_training_selected_f1_threshold") != LEGACY_THRESHOLD:
        raise ValueError("r92 threshold metadata changed")
    if state.get("embedding_weights") != "ResNet50_Weights.IMAGENET1K_V2" or state.get("embedding_dimensions") != 2048 or state.get("pca", {}).get("components_shape") != [32, 2048] or state.get("prototype", {}).get("shape") != [2048]:
        raise ValueError("r92 transform association changed")
    _npy(data["prototype.npy"], (2048,), "prototype")
    _npy(data["pca32_mean.npy"], (2048,), "PCA mean")
    _npy(data["pca32_components.npy"], (32, 2048), "PCA components")
    return data


@dataclass
class R92Predictor:
    booster: object
    order: tuple[str, ...]
    medians: object
    model_sha256: str

    def predict(self, rows: object) -> object:
        import numpy as np

        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            matrix = np.asarray([[row[name] for name in self.order] for row in rows], dtype=np.float32)
        else:
            matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 93 or np.isinf(matrix).any():
            raise ValueError("r92 input must be an N x 93 finite-or-NaN matrix")
        matrix = matrix.copy()
        rr, cc = np.where(np.isnan(matrix))
        matrix[rr, cc] = self.medians[cc]
        if not np.isfinite(matrix).all():
            raise ValueError("r92 imputation produced nonfinite values")
        prediction = np.asarray(self.booster.inplace_predict(matrix), dtype=np.float32)
        if prediction.shape != (matrix.shape[0],) or not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
            raise RuntimeError("r92 predictor returned invalid probabilities")
        return prediction


def load_model(path: Path) -> R92Predictor:
    import numpy as np
    import xgboost as xgb

    if xgb.__version__ != "2.1.1":
        raise ValueError("r92 sample requires XGBoost 2.1.1")
    data = _payloads(path)
    schema = _json(data["feature_schema.json"], "feature schema")
    imp = _json(data["imputer.json"], "imputer")
    order = tuple(schema["feature_order"])
    medians = np.asarray([imp["statistics"][name] for name in order], dtype=np.float32)
    if not np.isfinite(medians).all():
        raise ValueError("r92 fitted medians are nonfinite")
    booster = xgb.Booster()
    booster.load_model(bytearray(data["classifier.ubj"]))
    booster.set_param({"device": "cpu", "nthread": 4})
    if tuple(booster.feature_names or ()) != order or booster.num_features() != 93 or booster.num_boosted_rounds() != 796:
        raise ValueError("r92 classifier schema/tree count changed")
    if booster.attr("best_iteration") is not None:
        raise ValueError("r92 classifier has unexpected best_iteration semantics")
    return R92Predictor(booster, order, medians, CLASSIFIER_SHA256)


def read_feature_csv(path: Path, order: tuple[str, ...]) -> tuple[list[dict[str, str]], object]:
    import numpy as np

    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or not {"img_folder", "image", "id", *order} <= set(names):
            raise ValueError("CSV requires unique img_folder,image,id and all 93 ordered model fields")
        rows = list(reader)
    if not rows:
        raise ValueError("r92 input table is empty")
    keys = [(r["img_folder"], r["image"], r["id"]) for r in rows]
    if any(not all(k) for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("r92 input keys are empty or duplicated")
    try:
        matrix = np.asarray([[float(r[n]) if r[n] not in (None, "") else np.nan for n in order] for r in rows], dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("r92 input feature is not numeric") from exc
    return rows, matrix


def score_csv(model_path: Path, input_csv: Path, output: Path, threshold: float = PAPER_THRESHOLD) -> dict:
    if threshold not in (PAPER_THRESHOLD, LEGACY_THRESHOLD):
        raise ValueError("select the declared paper or legacy threshold explicitly")
    if Path(output).exists():
        raise FileExistsError("r92 output already exists; choose a new directory")
    model = load_model(model_path)
    rows, matrix = read_feature_csv(input_csv, model.order)
    p = model.predict(matrix)
    Path(output).mkdir(parents=True)
    try:
        scored = Path(output) / "detections.csv"
        with scored.open("w", newline="", encoding="utf-8") as stream:
            fields = ["img_folder", "image", "id", "xgb_p", "xgb_pred", "threshold", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "poly"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, value in zip(rows, p, strict=True):
                writer.writerow({"img_folder":row["img_folder"],"image":row["image"],"id":row["id"],"xgb_p":repr(float(value)),"xgb_pred":int(value >= threshold),"threshold":repr(threshold),"bbox_x":row.get("bbox_x", ""),"bbox_y":row.get("bbox_y", ""),"bbox_w":row.get("bbox_w", ""),"bbox_h":row.get("bbox_h", ""),"poly":row.get("poly", "")})
        receipt = {"schema":"compag-r92-score/v1","status":"PASS","model_sha256":model.model_sha256,"model_archive_sha256":ARCHIVE_SHA256,"input_sha256":_file_sha(Path(input_csv)),"detections_sha256":_file_sha(scored),"row_count":len(rows),"probability_dtype":str(p.dtype),"probability_shape":list(p.shape),"threshold":threshold,"threshold_role":"paper_example" if threshold==PAPER_THRESHOLD else "legacy_training_selected","execution_variant":"cpu_tabular_xgboost_2.1.1","scientific_accuracy_claim":False}
        (Path(output)/"SCORE_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise


def synthetic_sample(model_path: Path, output: Path, threshold: float = PAPER_THRESHOLD) -> dict:
    model = load_model(model_path)
    if Path(output).exists():
        raise FileExistsError("sample output exists")
    Path(output).mkdir(parents=True)
    fixture = Path(output)/"SYNTHETIC_FEATURES.csv"
    try:
        with fixture.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["img_folder","image","id",*model.order]);writer.writeheader()
            for i in range(3):
                values={name:repr(float(model.medians[j])) for j,name in enumerate(model.order)}
                for name in model.order:
                    if name.startswith("g_"): values[name]=""
                if i==1: values["area_norm"]="0.02"
                if i==2: values["area_norm"]="0.8";values["mean_a"]="130"
                writer.writerow({"img_folder":"SYNTHETIC","image":"SYNTHETIC_y00000x00000.jpg","id":str(i+1),**values})
        target=Path(output)/"scored"
        receipt=score_csv(model_path,fixture,target,threshold)
        (Path(output)/"SAMPLE_RESULT.json").write_text(json.dumps({"status":"PASS","kind":"SYNTHETIC_SOFTWARE_CHECK_ONLY","score_receipt":receipt},indent=2)+"\n")
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise


def cache_local(source: Path, cache: Path) -> Path:
    _payloads(source)
    target=Path(cache)/ARCHIVE_NAME
    Path(cache).mkdir(parents=True,exist_ok=True)
    if target.exists():
        _payloads(target)
        return target
    if Path(source).is_dir():
        raise ValueError("cache import requires the pinned ZIP, not a directory")
    with tempfile.NamedTemporaryFile(dir=cache,prefix=".r92-",delete=False) as tmp:
        temp=Path(tmp.name)
    try:
        shutil.copyfile(source,temp)
        _payloads(temp)
        os.replace(temp,target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def download(url: str, cache: Path) -> Path:
    """Opt-in HTTPS fetch; no release endpoint is built into the package."""
    parsed=urlsplit(url)
    if parsed.scheme!="https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("a reviewed HTTPS asset URL without credentials/fragment is required")
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    target=cache/ARCHIVE_NAME
    if target.exists():
        _payloads(target)
        return target
    req=urllib.request.Request(url,headers={"User-Agent":"compag-curation-r92-sample/1"})
    with urllib.request.urlopen(req,timeout=30) as response:
        if urlsplit(response.geturl()).scheme!="https":
            raise ValueError("r92 download redirected away from HTTPS")
        with tempfile.NamedTemporaryFile(dir=cache,prefix=".r92-",delete=False) as tmp:
            temp=Path(tmp.name)
            try:
                remaining=ARCHIVE_BYTES+1
                while remaining:
                    block=response.read(min(1024*1024,remaining))
                    if not block:break
                    tmp.write(block);remaining-=len(block)
            finally:
                tmp.flush()
    try:
        _payloads(temp)
        os.replace(temp,target)
    finally:
        temp.unlink(missing_ok=True)
    return target
