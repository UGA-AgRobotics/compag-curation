# Quick start — COMPAG 1.9.4rc10

1. Clone the repository. The four required program and example files are already included in `bundled-assets/`.
2. On Linux with a compatible NVIDIA GPU and Conda, open a terminal in this repository and run:

   ```bash
   python3 start_compag.py
   ```

3. The command opens a local browser page. It checks the included files automatically. Follow the installation step, then analyze an example card or upload your own JPG/PNG photo.
4. Open the candidate reviewer in another browser tab. After saving your decisions, return to the start page and stop its review session. Choose **Operational** XGBoost for a fresh fit without a holdout, or **Paper protocol** for the group-aware paper settings. Paper protocol requires individually reviewed decisions from at least six independent original photos and rejects bulk-accepted model predictions. The page shows what was trained.
5. If you have the existing CJ YOLO segmentation model, optionally attach it in the browser for box-based Prompt, Fusion, or Both inference on the next photo. The guide can prepare its separate pinned environment. This does not train YOLO. Analyze another photo using **Use the latest model trained from my previous review**, then repeat the review and XGBoost steps as needed.

If the browser does not open automatically, use the local address printed in the terminal. The basic operational terminal menu is `python3 start_compag.py --text`; the new Paper protocol and YOLO attachment choices are in the browser. For details and prerequisites, read [Start Here](START_HERE_EN.md). Full command-level steps remain in [FIRST_RUN_COMPLETE_ROUND_EN.md](FIRST_RUN_COMPLETE_ROUND_EN.md).

The quick-demo output basename must be printable ASCII; spaces are allowed. This does not restrict Unicode workspace or environment paths generally.
