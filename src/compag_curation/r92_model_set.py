"""Atomic activation and older-model selection for public project rounds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .r92_project_model import load_project_model, verify_snapshot


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def verify_model_set(root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    record = _read(root / "MODEL_SET.json")
    if record.get("schema") != "compag-r92-project-model-set/v1" or record.get("status") != "MODEL_SET_VERIFIED":
        raise ValueError("model set identity differs")
    snapshot = Path(record["snapshot"])
    if _sha(snapshot) != record["snapshot_sha256"]:
        raise ValueError("model set snapshot changed")
    verify_snapshot(snapshot)
    xgb = Path(record["xgb_bundle"])
    if _sha(xgb / "PROJECT_MODEL_MANIFEST.json") != record["xgb_manifest_sha256"]:
        raise ValueError("model set XGBoost manifest changed")
    load_project_model(xgb)
    manifest = _read(xgb / "PROJECT_MODEL_MANIFEST.json")
    if manifest["snapshot_sha256"] != record["snapshot_sha256"]:
        raise ValueError("model set classifier belongs to another dataset")
    native = Path(record["native_model"])
    if _sha(native) != record["native_model_sha256"]:
        raise ValueError("native transform archive changed")
    mode = record["yolo_mode"]
    if mode != "off":
        if record.get("yolo_status") not in {"YOLO_TRAINED", "YOLO_RETAINED", "YOLO_EXTERNAL_ATTESTED"}:
            raise ValueError("model set YOLO status differs from enabled mode")
        checkpoint=Path(record["yolo_checkpoint"])
        if checkpoint.is_symlink() or not checkpoint.is_file() or _sha(checkpoint) != record["yolo_checkpoint_sha256"]:
            raise ValueError("model set detector checkpoint changed")
        evidence_path=record.get("yolo_training_record") or record.get("yolo_attestation")
        evidence_sha=record.get("yolo_training_record_sha256") or record.get("yolo_attestation_sha256")
        if evidence_path is None or _sha(Path(evidence_path)) != evidence_sha:
            raise ValueError("model set detector attestation changed")
        evidence=_read(Path(evidence_path))
        if evidence.get("checkpoint_sha256") != record["yolo_checkpoint_sha256"] \
                or evidence.get("class_map") != {"0":"CJ"} or evidence.get("status") != "PASS":
            raise ValueError("model set detector attestation differs")
        if record["yolo_status"] == "YOLO_EXTERNAL_ATTESTED":
            from .r92_yolo import verify_segment_attestation
            if (record.get("yolo_external") is not True or record.get("yolo_attestation") is None
                    or record.get("yolo_training_record") is not None
                    or record.get("retained_from_round") is not None
                    or record.get("retained_model_set") is not None):
                raise ValueError("external detector model set has inconsistent provenance")
            verify_segment_attestation(evidence, checkpoint, record["yolo_checkpoint_sha256"])
        elif evidence.get("task") != "detect" or record.get("yolo_external") not in {None, False}:
            raise ValueError("trained or retained detector task differs")
        if record["yolo_status"] == "YOLO_TRAINED" and (record.get("yolo_training_record") is None or
                evidence.get("schema") != "compag-r92-yolo-project-model/v1"):
            raise ValueError("trained detector record differs")
        if record.get("yolo_status") == "YOLO_RETAINED":
            if record.get("yolo_attestation") is None or evidence.get("schema") != "compag-r92-yolo-checkpoint-attestation/v1":
                raise ValueError("retained detector attestation differs")
            retained = Path(record["retained_model_set"]).resolve(strict=True)
            if retained == root or _sha(retained / "MODEL_SET.json") != record["retained_model_set_sha256"]:
                raise ValueError("retained detector model-set identity changed")
            prior = verify_model_set(retained)
            if prior["round_id"] != record["retained_from_round"] or \
                    prior["yolo_checkpoint_sha256"] != record["yolo_checkpoint_sha256"]:
                raise ValueError("retained detector does not belong to named earlier round")
    elif record.get("yolo_checkpoint") is not None or record.get("yolo_status") != "YOLO_NOT_REQUESTED":
        raise ValueError("OFF model set must not carry a YOLO checkpoint")
    return record


def activate_model_set(workspace: Path, round_id: str, snapshot: Path, xgb_bundle: Path,
                       native_model: Path, output: Path, *, yolo_mode: str = "off",
                       yolo_checkpoint: Path | None = None, yolo_sha256: str | None = None,
                       yolo_training_record: Path | None = None,
                       yolo_attestation: Path | None = None,
                       yolo_external: bool = False,
                       retained_from_round: str | None = None,
                       retained_model_set: Path | None = None,
                       parent_model_set: Path | None = None) -> dict:
    if not round_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in round_id):
        raise ValueError("round ID must be a portable one-component name")
    if yolo_mode not in {"off", "prompt", "fusion", "both"}:
        raise ValueError("unsupported YOLO inference mode")
    workspace = Path(workspace).resolve(strict=True)
    snapshot, xgb_bundle, native_model, output = (Path(p).resolve() for p in
                                                 (snapshot, xgb_bundle, native_model, output))
    if output.exists() or output == workspace or workspace not in output.parents:
        raise ValueError("new model set must be under the existing workspace")
    verify_snapshot(snapshot)
    load_project_model(xgb_bundle)
    manifest = _read(xgb_bundle / "PROJECT_MODEL_MANIFEST.json")
    if manifest["snapshot_sha256"] != _sha(snapshot):
        raise ValueError("project XGBoost model and snapshot differ")
    if _sha(native_model) != manifest["native_model_sha256"]:
        raise ValueError("native feature-state archive differs")
    if yolo_mode == "off":
        if yolo_checkpoint is not None or yolo_sha256 is not None or yolo_training_record is not None or yolo_attestation is not None or yolo_external or retained_from_round is not None or retained_model_set is not None:
            raise ValueError("YOLO disabled model set has unexpected detector arguments")
        yolo_status = "YOLO_NOT_REQUESTED"
    else:
        if yolo_checkpoint is None or yolo_sha256 is None:
            raise ValueError("selected YOLO inference mode requires explicit verified checkpoint")
        if sum((yolo_training_record is not None, retained_from_round is not None, yolo_external)) != 1:
            raise ValueError("detector must be newly trained, retained, or externally attested")
        if Path(yolo_checkpoint).is_symlink() or not Path(yolo_checkpoint).is_file() or _sha(Path(yolo_checkpoint)) != yolo_sha256:
            raise ValueError("requested YOLO checkpoint missing or changed")
        if yolo_external:
            if yolo_attestation is None or retained_model_set is not None:
                raise ValueError("external detector requires its attestation and no retained model set")
            from .r92_yolo import verify_segment_attestation
            record = _read(yolo_attestation)
            verify_segment_attestation(record, Path(yolo_checkpoint), yolo_sha256)
            yolo_status = "YOLO_EXTERNAL_ATTESTED"
        elif yolo_training_record is not None:
            if yolo_attestation is not None or retained_model_set is not None:
                raise ValueError("new detector training record does not need a retained checkpoint attestation")
            record = _read(yolo_training_record)
            if record.get("schema") != "compag-r92-yolo-project-model/v1" or record.get("status") != "PASS" \
                    or record.get("checkpoint_sha256") != yolo_sha256 or record.get("class_map") != {"0":"CJ"}:
                raise ValueError("YOLO training record and selected checkpoint differ")
            yolo_status = "YOLO_TRAINED"
        else:
            if yolo_attestation is None or retained_model_set is None:
                raise ValueError("retained detector requires attestation and the verified earlier model set")
            record = _read(yolo_attestation)
            if record.get("schema") != "compag-r92-yolo-checkpoint-attestation/v1" or record.get("status") != "PASS" \
                    or record.get("checkpoint_sha256") != yolo_sha256 or record.get("class_map") != {"0":"CJ"}:
                raise ValueError("retained detector attestation differs")
            prior_detector = verify_model_set(retained_model_set)
            if prior_detector["round_id"] != retained_from_round or \
                    prior_detector["yolo_checkpoint_sha256"] != yolo_sha256:
                raise ValueError("retained detector does not belong to named earlier round")
            yolo_status = "YOLO_RETAINED"
    parent_sha = None
    if parent_model_set is not None:
        verify_model_set(parent_model_set)
        parent_sha = _sha(Path(parent_model_set) / "MODEL_SET.json")
    record = {"schema": "compag-r92-project-model-set/v1", "status": "MODEL_SET_VERIFIED",
        "ready_for_next_inference": True, "round_id": round_id,
        "parent_model_set_sha256": parent_sha, "snapshot": str(snapshot),
        "snapshot_sha256": _sha(snapshot), "review_origin": _read(snapshot)["review_origin"],
        "xgb_bundle": str(xgb_bundle),
        "xgb_manifest_sha256": _sha(xgb_bundle / "PROJECT_MODEL_MANIFEST.json"),
        "native_model": str(native_model), "native_model_sha256": _sha(native_model),
        "yolo_mode": yolo_mode, "yolo_status": yolo_status,
        "yolo_external": bool(yolo_external),
        "yolo_checkpoint": str(Path(yolo_checkpoint).resolve()) if yolo_checkpoint else None,
        "yolo_checkpoint_sha256": yolo_sha256,
        "yolo_training_record_sha256": _sha(yolo_training_record) if yolo_training_record else None,
        "yolo_training_record": str(Path(yolo_training_record).resolve()) if yolo_training_record else None,
        "yolo_attestation_sha256": _sha(yolo_attestation) if yolo_attestation else None,
        "yolo_attestation": str(Path(yolo_attestation).resolve()) if yolo_attestation else None,
        "retained_from_round": retained_from_round,
        "retained_model_set": str(Path(retained_model_set).resolve()) if retained_model_set else None,
        "retained_model_set_sha256": _sha(Path(retained_model_set) / "MODEL_SET.json") if retained_model_set else None}
    output.mkdir(parents=True)
    try:
        (output / "MODEL_SET.json").write_text(json.dumps(record, indent=2) + "\n")
        verify_model_set(output)
        pointer = workspace / "LATEST_MODEL_SET.json"
        temp = workspace / ".LATEST_MODEL_SET.tmp"
        if temp.exists():
            raise FileExistsError("activation temporary pointer already exists")
        temp.write_text(json.dumps({"schema":"compag-r92-active-model-set/v1",
                                    "path":str(output),"model_set_sha256":_sha(output/"MODEL_SET.json")}, indent=2)+"\n")
        os.replace(temp, pointer)
        return record
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise


def infer_model_set(model_set: Path, image_project: Path, asset_root: Path, output: Path,
                    *, yolo_boxes: Path | None = None,
                    checkpoint_dir: Path | None = None) -> dict:
    from .r92_live import run_full
    record = verify_model_set(model_set)
    return run_full(image_project, Path(record["native_model"]), asset_root, output,
        project_model=Path(record["xgb_bundle"]), yolo_mode=record["yolo_mode"],
        yolo_checkpoint=Path(record["yolo_checkpoint"]) if record["yolo_checkpoint"] else None,
        yolo_sha256=record["yolo_checkpoint_sha256"],yolo_boxes=yolo_boxes,
        checkpoint_dir=checkpoint_dir)


def complete_xgb_round(workspace: Path, round_id: str, state: Path, inference: Path,
                       native_model: Path, output: Path, *, selection: Path | None = None,
                       parent_snapshot: Path | None = None,
                       parent_model_set: Path | None = None,
                       review_origin: str = "human", estimators: int = 100,
                       scope: str = "reviewed", correction_revision: str | None = None) -> dict:
    """One user command after review: seal, fit, verify, activate.

    Stages are individually immutable. A failed stage leaves prior stage artifacts
    for inspection; a new output revision is required after changing inputs.
    """
    from .r92_project_model import finalize_review, train_small_data

    workspace, output = Path(workspace).resolve(strict=True), Path(output).resolve()
    if output.exists() or workspace not in output.parents:
        raise ValueError("new round output must be under the workspace")
    snapshot = output / "training_snapshot.json"
    xgb = output / "xgb_model"
    model_set = output / "model_set"
    def stage(name: str, **details) -> None:
        target = output / "ROUND_STATE.json"
        temp = output / ".ROUND_STATE.tmp"
        temp.write_text(json.dumps({"schema":"compag-r92-round-state/v1",
            "round_id":round_id,"state":name,**details},indent=2)+"\n")
        os.replace(temp,target)
    finalized = finalize_review(state,inference,native_model,snapshot,
        selection=selection,parent_snapshot=parent_snapshot,
        review_origin=review_origin,confirm_review_complete=True,
        scope=scope,correction_revision=correction_revision)
    stage("DATASET_SNAPSHOT_READY", snapshot_sha256=_sha(snapshot),
          review_origin=review_origin, yolo_status="YOLO_NOT_REQUESTED")
    fitted = train_small_data(snapshot,native_model,xgb,n_estimators=estimators)
    stage("XGB_TRAINED", snapshot_sha256=_sha(snapshot),
          xgb_manifest_sha256=_sha(xgb/"PROJECT_MODEL_MANIFEST.json"),
          yolo_status="YOLO_NOT_REQUESTED")
    activated = activate_model_set(workspace,round_id,snapshot,xgb,native_model,model_set,
        parent_model_set=parent_model_set)
    stage("READY_FOR_NEXT_INFERENCE", snapshot_sha256=_sha(snapshot),
          xgb_manifest_sha256=_sha(xgb/"PROJECT_MODEL_MANIFEST.json"),
          model_set_sha256=_sha(model_set/"MODEL_SET.json"),
          yolo_status=activated["yolo_status"])
    return {"schema":"compag-r92-complete-xgb-round/v1","status":"READY_FOR_NEXT_INFERENCE",
            "round_id":round_id,"finalized":finalized,"trained":fitted,
            "model_set":str(model_set),"model_set_sha256":_sha(model_set/"MODEL_SET.json"),
            "yolo_status":activated["yolo_status"]}
