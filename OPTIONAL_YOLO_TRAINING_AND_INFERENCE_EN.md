> For the separately supplied one-class CJ segmentation checkpoint (Ultralytics 8.4.26), see [EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md](EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md). The training instructions below remain for the detection checkpoint.

# Current r92 optional YOLO inference

The browser and CLI retain the internal `off|prompt|fusion|both` mode values.
`off` is the canonical paper `xgb_recall` path with YOLO disabled and
XGBoost-only decisions. `prompt` adds YOLO-box prompts to AMG
(`sam2_policy=both`) but keeps `det_policy=xgb`. `fusion` retains AMG
(`sam2_policy=auto`) and uses `det_policy=hybrid`. `both` combines
`sam2_policy=both` with `det_policy=hybrid`. The `prompt` mode is not
`sam2_policy=prompt`, which would use prompted masks alone.

For `det_policy=hybrid`, a YOLO match requires confidence at least 0.20 and
candidate-box IoU at least 0.60. The paper's weighted fusion uses a 0.50
decision threshold and `hybrid_yolo_bias=0.80` when XGBoost is high and YOLO
is low. With YOLO enabled, `det_missing=reject` means a candidate without a
valid match has a negative final decision regardless of its XGBoost score.
Its review score `p_ui` falls back to `p_xgb` for uncertainty ranking; when
YOLO is disabled, XGBoost alone decides at 0.50. The external segmentation
checkpoint and current browser steps are documented in
[EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md](EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md).

# Optional CJ detector in rc6 (historical detection path)

The default project path is XGBoost only. `train-yolo` is an independent optional operation. Training a detector never turns it on for inference; `activate-model-set --yolo-mode off|prompt|fusion|both` is a separate choice. A retained older detector needs an explicit checkpoint hash, `--retained-from-round`, `--retained-model-set` for the verified source round, and a fresh optional-environment checkpoint attestation; a new one needs its training record. An omitted or changed requested checkpoint fails before SAM2 inference.

## One-class detection data and QA

The detector task is `detect`, class `0=CJ`. XGBoost labels are `0=non-CJ, 1=CJ`; COCO categories are `1=CJ, 2=non-CJ`. These numbers are not interchangeable. Non-CJ proposals are not a second detector class.

`r92 prepare-yolo` consumes a `compag-r92-yolo-complete-tile-qa/v1` JSON bound to the exact prepared image-project receipt. Each unit needs its prepared tile name/hash, `split_role` (`train`, `val`, or `test`), `complete_for: "all_CJ_targets_in_512x512_tile"`, a nonempty `certified_by`, and an explicit `targets` list. Each target needs a stable `physical_object_id` and a valid `bbox_xyxy` in that tile's 512-pixel coordinates. An empty target list is permitted only with the same explicit completeness certification. The converter rejects partial/incomplete units, invalid boxes, close duplicate boxes, changed tiles, and cards split across train/validation/test. Independent train and validation cards are mandatory. It writes a sealed image/label dataset and `DATASET_MANIFEST.json`.

An active-learning shortlist, skipped candidate, or single non-target decision does not certify the rest of a tile as target-free. This implementation does not offer a partial-label ignore loss. Uncertain boxes must be resolved during complete-tile QA; omitting one and leaving its pixels as background is invalid. YOLO loss does not use the XGBoost 0.4 review factor. Externally supplied detector boxes are a separate annotation channel and do not alter old XGBoost features. If candidate geometry is edited, old feature rows must not be reused.

Example QA shape (replace hashes, tile names, and annotations with inspected real values):

```json
{
  "schema": "compag-r92-yolo-complete-tile-qa/v1",
  "target_class": "CJ",
  "project_receipt_sha256": "<64 lowercase hex digits>",
  "annotation_origin": "human",
  "units": [
    {
      "tile": "IMG_9459_y00000x00000.jpg",
      "tile_sha256": "<64 lowercase hex digits>",
      "split_role": "train",
      "complete_for": "all_CJ_targets_in_512x512_tile",
      "certified_by": "reviewer identity",
      "targets": [{"physical_object_id": "card-object-001", "bbox_xyxy": [120, 100, 180, 170]}]
    }
  ]
}
```

`r92 train-yolo` verifies that sealed dataset, an explicit local one-class CJ checkpoint (fine-tune) or one-class YOLO YAML (random initialization), and Ultralytics 8.3.221. It copies the dataset into a named private `.input-stage` because Ultralytics writes cache files during fitting; the sealed QA dataset remains unchanged. It invokes the retained local trainer, saves `best.pt`, reloads it, and predicts on a validation image. The record includes task/class map, data and initial-weight hashes, source-card split, requested epochs/batch/size/device/seed, the backend's auto-optimizer policy and saved-args hash, augmentation settings, checkpoint-selection rule, dependency versions, hardware, and prediction probe. A one-epoch run demonstrates optimizer operation only.

A requested fit that raises an error writes a sibling `<output>.YOLO_FAILED.json` when the directory is writable. Any valid XGBoost bundle remains intact; no combined model set is activated from that failed fit. Choose a new training output and explicitly select retry, YOLO OFF, or an attested retained detector. A new round fit is not optimizer-state resume.

## Inference routing in the rc6 detection path

`r92 yolo-predict-boxes` executes over **every** prepared tile on CPU in the optional Ultralytics environment, writing a tile/weight/hash-bound box receipt. `infer-model-set --yolo-boxes` verifies that receipt in the base science-gpu environment without importing Ultralytics. `off` does not import or load YOLO. `prompt` passes valid detector boxes to SAM2's box predictor (the retained 2% box padding), then merges those masks with canonical AMG proposals; mask hashes and source provenance change with the candidate pool. `fusion` keeps AMG proposals and matches a detector box by IoU at least 0.60, then applies the retained Only_codes reviewer hybrid rule. `both` enables both paths. The output keeps raw `xgb_p`, detector confidence/IoU, `fused_p`, and final decision separate. A valid detector with no matching box is recorded as no support; missing/corrupt requested weights are errors. Full output includes all per-tile YOLO boxes, their hash, checkpoint hash, mode, and proposal source.

## Optional dependency and distribution scope

The rc6 public wheel does not bundle Ultralytics, a detector checkpoint, training images, or synthetic fixture weights. The optional backend was locally tested with Ultralytics 8.3.221 in a separate environment in the producer's prior evidence; this audit did not rerun a YOLO fit. `r92 verify-yolo-checkpoint` supplies an attestation when retaining an older detector; newly trained weights use their saved training record. The [pinned Ultralytics v8.3.221 license](https://github.com/ultralytics/ultralytics/blob/v8.3.221/LICENSE) is AGPL-3.0, and [Ultralytics describes a separate Enterprise option](https://github.com/ultralytics/ultralytics/blob/v8.3.221/README.md). Review the terms for the intended distribution or service integration before publishing a combined detector-enabled product. This local candidate grants no redistribution rights to an external checkpoint or the user's training data.
