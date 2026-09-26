# Start COMPAG after getting it from GitHub

This is the first-run guide for the **Linux science-GPU** workflow. Start with one command, then follow the steps in your **local web browser**. You do not need to type the long installation, inference, review, or training commands.

## Before you start

1. Clone the COMPAG repository. Open a terminal in its top-level folder, where `start_compag.py` is located.
2. The four required files are already in this clone's `bundled-assets/` folder: the COMPAG wheel, its checksum sidecar, the native model ZIP, and the ten-photo ZIP. Leave these files where they are. You do not need to download them separately or enter their paths.
3. Use a Linux x86-64 computer with Python 3.10 or newer, a working NVIDIA CUDA GPU, Conda available as `conda`, internet access for the pinned dependencies and official model assets, and **at least 15 GiB of free disk space**. A normal user account is sufficient.

## Open the local browser guide

```bash
python3 start_compag.py
```

The command opens the guide in your web browser. If the browser does not open automatically, copy the local address printed in the terminal into a browser on the **same computer**. Keep the terminal running while you use the guide. The page is served locally; it is not an online COMPAG account or a public upload service.

If you cannot use the browser guide, a terminal menu offers the basic operational workflow. The Paper protocol XGBoost choice and existing YOLO attachment are available in the browser guide:

```bash
python3 start_compag.py --text
```

## Follow the steps on the page

1. **Check included files.** The guide finds the four files in `bundled-assets/` automatically and checks their fixed hashes before installing anything. No upload or file path is needed.
2. **Install and prepare.** Choose an installation location and a private work folder **outside the cloned repository**. The guide installs the pinned science-GPU environment, checks it, downloads and verifies the official SAM2/ResNet assets, and prepares the ten example cards. This can take time; follow progress on the page and keep the terminal running. An existing verified installation and work folder can be reused.
3. **Analyze a card.** Choose one of the supplied example cards or upload one of your own JPG/PNG photos. The guide prepares the image and runs full-card inference with the existing COMPAG engine. Your uploaded photo stays in your private local work folder. Completed tiles from an interrupted attempt can be reused when you resume that session.
4. **Review candidates.** Open the existing local reviewer from the guide; it appears in another browser tab. Inspect candidates and save your human decisions. You can delete unwanted masks so they do not enter the pool. **Accept remaining model predictions** records the remaining model decisions in bulk; those decisions are not independent human labels. Your review state remains in the work folder, so you can return later.
5. **Train XGBoost for the next round.** When review is finished, return to the guide and stop its review session if it is still running. Check the saved-decision summary and choose a training policy before selecting **Train XGBoost**. **Operational** performs a real fresh fit on eligible cumulative rows with a fixed 100-round recipe and no holdout or cross-validation. It needs both CJ and non-CJ labels; a result based on bulk acceptance mainly learns repeated model predictions. **Paper protocol** instead applies the documented group-aware XGBoost method to new review data. It rejects bulk-accepted predictions and requires individually reviewed decisions from at least six independent original cards, both classes, and valid group-safe training, validation, and held-out splits. If those conditions fail, the guide explains the issue before fitting or changing your latest model. Unreviewed, skipped, and deleted candidates do not contribute to the reviewed-decision training scope. The page shows the saved model's policy, row count, training evidence, and classifier SHA-256 rather than treating a software pass as a scientific result.
6. **Optionally add an existing YOLO checkpoint.** After an XGBoost round, choose the folder of your existing one-class CJ segmentation YOLO package on this computer. That folder must contain `MODEL_MANIFEST.json` and `weights/best.pt`. Select Off, Prompt, Fusion, or Both for the next analysis. Leave the advanced YOLO Python field blank to let the guide prepare a separate hash-pinned Ultralytics 8.4.26 environment without changing the science-GPU installation. COMPAG verifies the package and checkpoint, then uses its boxes and confidences on every image tile. Prompt supplies boxes to SAM2; Fusion combines YOLO support with XGBoost decisions; Both enables both routes. This step **does not train YOLO**. The external package is not one of the four included files.
7. **Continue the cycle.** Choose another photo and keep **Use the latest model trained from my previous review** selected to use the current XGBoost model and any YOLO setting attached to it. Review its candidates, then explicitly train XGBoost again when you are ready. Later XGBoost retraining carries forward the selected, verified YOLO inference setting. Training is never triggered simply by closing the review page.

**Starting a clean paper-protocol lineage:** A prior bulk-accepted round remains in the cumulative training snapshot and will make paper preflight fail even after later individual reviews. To start without that parent, clear **Use the latest model trained from my previous review** for a new card, individually review examples of both classes, and use Operational training for its first model. Then analyze additional distinct original cards with the latest model selected, individually review them, and train each continuation. Select Paper protocol only after the cumulative lineage has enough independent cards, both classes, and valid group-safe splits. A new private work folder is another way to start a separate lineage.

## Return to your work

Run `python3 start_compag.py` again from the same clone and use the same work folder. Your saved sessions, review decisions, and trained models are there. The guide can reopen a review or resume interrupted inference. Keep the work folder if you move or refresh the source clone.

For the full command-level workflow and scientific scope, see [FIRST_RUN_COMPLETE_ROUND_EN.md](FIRST_RUN_COMPLETE_ROUND_EN.md). The external YOLO segmentation checkpoint is optional, supplied separately, and documented in [EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md](EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md). Newly uploaded photos receive distinct saved names so separate originals can be grouped correctly. Older review snapshots that used the same `original.jpg` name for different uploads cannot be silently reidentified; use new sessions and reviews if you need those photos in paper-protocol training. The paper-protocol option applies the paper's training settings to new eligible review data; it does not reproduce the historical r92 fit or its reported accuracy. A successful software run or the supplied examples do not establish biological accuracy.
