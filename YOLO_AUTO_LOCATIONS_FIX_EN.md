# R5.3 launcher hotfix: automatic YOLO locations

The browser accepts an empty YOLO package field and asks the launcher to find
the existing CJ segmentation package. Saved paths are tried first, followed by
the workspace, source folder, two nearby parent folders, and their immediate
subfolders (including `models/`). This is a bounded local search. On another
computer, place the separately supplied YOLO package under
`COMPAG_Workspace/models/` for automatic discovery.

The package must contain `MODEL_MANIFEST.json` and `weights/best.pt`. Discovery
does not activate a detector. Select the intended YOLO mode and click **Apply
YOLO setting** once. The existing checksum and runtime checks still run before
activation. The selected package and Python are then remembered, including
across XGBoost retraining, restarting the guide, and temporarily disabling YOLO.
If different checkpoints are found with no saved identity, choose one once.
The launcher never silently replaces a saved checkpoint with another model.

The next photo uses the attached setting when **Use the latest model trained
from my previous review** is selected. An unapplied mode selection is flagged
before a new analysis starts. A new action clears earlier browser validation
errors, and the running analysis reports its YOLO mode.

Restart the local guide to load this change after the current operation or
review has finished. Run `python3 start_compag.py` from this folder again and
use the newly opened page. Existing R5.3 installations need no wheel reinstall
for this launcher-only hotfix. Existing sessions retain their recorded mode;
enabling YOLO does not retroactively modify an earlier analysis.

Scientific source, model weights, hybrid equations, thresholds, training
recipes, and review behavior are unchanged by this hotfix. The four bundled
startup files are byte-identical to the original R5.3 package. External YOLO
weights are not included in the public source archive.

Regression check: `python3 -B tools/test_easy_start_yolo.py -v`. The suite uses
synthetic fixtures and mocked scientific subprocesses; it tests discovery,
checksum refusal, activation persistence, next-session and retraining carry,
and browser events (the latter requires Node). It does not claim a new
full-card YOLO/SAM2 inference run or biological validation.
