"""Read-only installed-wheel acceptance; keeps evidence levels explicit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--native-model",type=Path,required=True)
    parser.add_argument("--inference",type=Path,required=True)
    parser.add_argument("--project-model",type=Path,required=True)
    parser.add_argument("--model-set",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from compag_curation import r92_sample
    from compag_curation.r92_project_model import load_project_model
    from compag_curation.r92_model_set import verify_model_set
    from compag_curation.r92_review import _full_inference_binding

    version=importlib.metadata.version("compag-curation")
    if version!="1.9.4rc5":
        raise RuntimeError("this runner requires installed rc5 distribution")
    r92_sample._payloads(args.native_model)
    full=json.loads((args.inference/"FULL_INFERENCE_RECEIPT.json").read_text())
    project=Path(full["project_path"])
    scores=args.inference/"scores"/"detections.csv"
    binding=_full_inference_binding(scores,project/"tiles")
    if binding is None or binding["tile_count"]!=full["expected_tile_count"]:
        raise RuntimeError("full-card inference binding failed")
    predictor=load_project_model(args.project_model)
    model_set=verify_model_set(args.model_set)
    evidence={"schema":"compag-r92-rc5-installed-acceptance/v1","status":"PASS",
        "evidence_level":"installed-wheel software acceptance; no human-review or scientific-performance claim",
        "installed_distribution_version":version,
        "installed_package_path":str(Path(__import__("compag_curation").__file__).resolve()),
        "native_model_sha256":sha(args.native_model),
        "full_inference_receipt_sha256":sha(args.inference/"FULL_INFERENCE_RECEIPT.json"),
        "full_card_tile_count":binding["tile_count"],
        "full_card_candidate_count":full["candidate_count"],
        "project_classifier_sha256":predictor.model_sha256,
        "model_set_sha256":sha(args.model_set/"MODEL_SET.json"),
        "model_set_round_id":model_set["round_id"],
        "review_origin":model_set["review_origin"]}
    if args.output.exists():
        raise FileExistsError("acceptance output already exists")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(evidence,sort_keys=True))


if __name__=="__main__":
    main()
