# COMPAG Curation 1.9.4rc10 — start here

The latest launcher update adds **Export COCO ZIP** for saved analyses, **Stop analysis / Stop training**, and explicit Paper protocol readiness checks. Existing R5.3 users only need to restart the guide. See [export and stop instructions](EXPORT_STOP_UPDATE.md).

After cloning this repository, run **`python3 start_compag.py`** from this folder. It opens a **local page in your web browser** and walks you through checking the four files included in `bundled-assets/`, installing the pinned GPU environment, analyzing a card, reviewing its candidates, and explicitly training XGBoost for the next card. The page also lets you attach an existing one-class YOLO segmentation checkpoint for optional detector support during the next analysis. You do not need a separate Release download or a path for the four included files. The terminal menu remains available with `python3 start_compag.py --text`. See [Start Here](START_HERE_EN.md) or [راهنمای فارسی](START_HERE_FA.md).

This local source revision retains the browser training-policy choice, guarded paper-settings XGBoost training helper, and automatic setup and activation of an existing YOLO segmentation checkpoint for inference. The checkpoint is not retrained by this browser workflow. R5.3 aligns the optional XGBoost/YOLO hybrid decision and browser mode names with the paper; the canonical YOLO-disabled run remains XGBoost-only. R5.4 advances the application to 1.9.4rc10 and rebuilds its matching wheel; rerun setup to install the new version. See [current R5.4 release notes](PUBLIC_RELEASE_NOTES_RC10_R5_4.md) and the retained [R5.3 history](PUBLIC_RELEASE_NOTES_RC9_R5_3.md).

## Release and research details

This is a versioned software companion containing retained scientific source, documented release adaptations and the identified r92 asset. It does not claim that the new launcher/export/Stop code generated historical paper results.

This is packaging revision `rc10-repair-r5.4` of application 1.9.4rc10. The optional, user-supplied one-class CJ YOLO segmentation checkpoint (Ultralytics 8.4.26) remains an external local asset. The browser guide can prepare a separate, hash-pinned YOLO environment, verify that checkpoint, and use its boxes and confidences in SAM2 prompting and XGBoost/YOLO fusion. In the paper hybrid policy, a valid YOLO match requires confidence at least 0.20 and box overlap at least 0.60. A candidate without valid YOLO support is rejected by the hybrid decision, while its review score uses the raw XGBoost probability. YOLO instance masks are verified but not inserted as proposals. The rc8 right-button reviewer pan and R5.2 review zoom continuity remain available. This staged repository includes the matching wheel and sidecar, immutable native r92 model ZIP, and exactly ten original full-card photos in a separate ZIP under `bundled-assets/`. The browser guide checks these included files automatically. No live URL, tag, or repository slug is asserted.

The author selected MIT for author-created application code and CC BY 4.0 for author-created documentation, the exact model, and the selected ten photographs. See [LICENSES.md](LICENSES.md), [the asset addendum](ASSET_LICENSES/PUBLIC_ASSET_SCOPE_EN.md), and [the photo manifest](PUBLIC_PHOTO_MANIFEST.json). Rights-holder and institutional/coauthor approvals are not attested. No other research data, images, annotations, logs, or newly fitted model weights are in this public set.

## Prepare all ten included example cards

The start guide handles these commands for newcomers. Advanced users can use the Python 3.12 science-GPU environment documented in [FIRST_RUN_COMPLETE_ROUND_EN.md](FIRST_RUN_COMPLETE_ROUND_EN.md), with Pillow 12.3.0. From a clone containing `bundled-assets/`:

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

The helper checks exact hashes and ZIP membership, decodes every JPEG, and writes `EXAMPLE_INPUTS.json` plus stable `cards/IMG_####/` directories for all ten cards. It rejects a pre-existing output. The first/next-card walkthrough in the complete-round guide uses two distinct cards; all ten remain available for the same full-card workflow. All candidate proposals can be reviewed, deleted or bulk accepted as documented. The browser offers two explicit XGBoost choices: **Operational** fits all eligible decisions with no holdout or cross-validation; **Paper protocol** uses group-safe search and validation on newly reviewed cards and rejects bulk-accepted predictions. It requires at least six independent original-card groups, both classes, valid train/heldout groups and five valid training folds. A trained model is not evidence of biological accuracy. The optional browser YOLO path uses a previously trained checkpoint for inference; it does not train YOLO. The synthetic `r92 sample` check does not constitute biological validation or genuine human review.

The ten images are author-selected public examples. Their membership in any paper ten-card audit has not been established. The original r92 model ZIP remains byte-identical. This is a local staged candidate; no remote release or no-login test has occurred.

`CAPABILITY_STATUS.json` retains the dated rc6 capability record for historical context. The paper hybrid update is recorded in [PUBLIC_RELEASE_NOTES_RC9_R5_3.md](PUBLIC_RELEASE_NOTES_RC9_R5_3.md); the reviewer zoom fix is recorded in [PUBLIC_RELEASE_NOTES_RC9_R5_2.md](PUBLIC_RELEASE_NOTES_RC9_R5_2.md); the Python 3.10 browser fix is recorded in [PUBLIC_RELEASE_NOTES_RC9_R5_1.md](PUBLIC_RELEASE_NOTES_RC9_R5_1.md). The R5 workflow is documented in [PUBLIC_RELEASE_NOTES_RC9_R5.md](PUBLIC_RELEASE_NOTES_RC9_R5.md), and the original YOLO adapter in [PUBLIC_RELEASE_NOTES_RC9.md](PUBLIC_RELEASE_NOTES_RC9.md).

For the external YOLO checkpoint and next-card pipeline, see [EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md](EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md). The external weight is not a GitHub Release asset.
