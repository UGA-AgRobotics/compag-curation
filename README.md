# COMPAG Curation

Software companion to our COMPAG paper on human-in-the-loop pest monitoring with sticky-trap cards. You analyze a card, review the candidate detections, and train an XGBoost model that is then used on the next card.

## Quick start

After cloning, run this from the repository folder:

```bash
python3 start_compag.py
```

This opens a local page in your browser that walks you through:

1. checking the four files in `bundled-assets/`
2. installing the pinned GPU environment
3. analyzing a card
4. reviewing its candidates
5. training XGBoost for the next card

Everything you need is already in `bundled-assets/`, so there's nothing extra to download and no paths to set. If you have a one-class YOLO segmentation checkpoint, you can attach it on the same page and it will be used as detector support in the next analysis.

If you'd rather stay in the terminal, use `python3 start_compag.py --text`.

## Features

- guided setup and a complete review round in the browser
- two XGBoost training modes (see below), with readiness checks before training in Paper protocol mode
- a guarded helper for training XGBoost with the paper settings
- **Stop analysis** and **Stop training** buttons
- **Export COCO ZIP** for saved analyses
- optional YOLO support using an existing checkpoint
- zoom and right-button panning in the reviewer

## What's in the repository

- the research source code
- the application wheel and its sidecar
- the inference model ZIP, byte-identical to the original
- a ZIP with ten full-card example photos

Nothing else from the project is included: no other images, annotations, logs, or trained weights. The YOLO checkpoint isn't included either; you need to supply your own.

The launcher, COCO export, and Stop buttons were added after the paper, so they weren't used to produce the results reported there.

## Training XGBoost

The browser offers two training modes:

- **Operational** fits on all eligible review decisions. There's no holdout and no cross-validation.
- **Paper protocol** uses group-safe search and validation on newly reviewed cards, and doesn't accept bulk-accepted predictions. It needs at least six independent original-card groups, both classes, valid train/held-out groups, and five valid training folds.

Keep in mind that having a trained model doesn't tell you anything about biological accuracy on its own.

## Optional YOLO support

You can plug in your own one-class CJ YOLO segmentation checkpoint trained with Ultralytics. The guide sets up a separate, hash-pinned environment for it, verifies the checkpoint, and then uses its boxes and confidences for SAM2 prompting and for XGBoost/YOLO fusion. The checkpoint is only used for inference; this workflow doesn't retrain YOLO.

The hybrid decision follows the paper:

- A YOLO match counts only if confidence is at least 0.20 and box overlap is at least 0.60.
- Candidates without a valid YOLO match are rejected by the hybrid decision. Their review score still shows the raw XGBoost probability.
- YOLO instance masks are checked but aren't added as proposals.

With YOLO turned off, the run is XGBoost-only, which is the canonical setup. For setup details and the next-card pipeline, see [EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md](EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md).

## Preparing the ten example cards by hand

The start guide runs these steps for you. If you want to do it yourself, set up the science-GPU environment described in [FIRST_RUN_COMPLETE_ROUND_EN.md](FIRST_RUN_COMPLETE_ROUND_EN.md), then run:

```bash
PREFIX=/absolute/path/to/installed-science-gpu-env
SOURCE=/absolute/path/to/cloned-COMPAG-repository
ASSETS="$SOURCE/bundled-assets"
EXAMPLE=/absolute/path/to/new-r92-example-inputs
"$PREFIX/bin/python" "$SOURCE/tools/prepare_r92_public_example.py" \
  --model-zip "$ASSETS/COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip" \
  --photos-zip "$ASSETS/COMPAG_R92_TEN_PHOTO_INPUTS_RC6_R3_20260924.zip" \
  --output "$EXAMPLE"
MODEL="$EXAMPLE/model/COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip"
CARD1="$EXAMPLE/cards/IMG_9317"
CARD2="$EXAMPLE/cards/IMG_9319"
"$PREFIX/bin/compag-curation" r92 verify --model "$MODEL"
```

The helper checks file hashes and ZIP contents, makes sure every JPEG decodes, and writes `EXAMPLE_INPUTS.json` along with one `cards/IMG_####/` folder per card. It won't write into an output folder that already exists.

The walkthrough in the complete-round guide uses two different cards (a first card and a next card), but all ten work the same way. Candidate proposals can be reviewed one by one, deleted, or bulk-accepted.

The included sample check runs on synthetic data, so treat it as a technical check only. It isn't a biological validation, and it doesn't replace real human review.

## About the example photos

The ten photos were picked by the author as public examples. We haven't checked whether they overlap with the ten cards audited in the paper.

## License

Application code is MIT. The documentation, the included model, and the ten example photos are CC BY 4.0. See [LICENSES.md](LICENSES.md), the asset addendum, and the photo manifest.