# COMPAG Curation 1.9.4rc9 — bundled browser onboarding revision R4

Run `python3 start_compag.py` from the cloned GitHub repository. The local browser guide now finds and verifies its four required files directly in `bundled-assets/`: the matching COMPAG wheel and checksum sidecar, the immutable native r92 model ZIP, and the ten-photo ZIP. These are ordinary repository files. There is no separate Release download, browser upload, or asset-folder path to provide.

The remaining steps are the same browser-guided workflow: install and verify the pinned science-GPU environment, download and verify the official SAM2 and ResNet assets, prepare the ten example cards, analyze an example or personal JPG/PNG photo, save decisions in the existing reviewer, explicitly train XGBoost from eligible decisions, and analyze the next photo with the latest trained model selected by default. Interrupted inference can resume. Work sessions, personal photos, review decisions, and newly fitted models remain in a private local work folder outside the repository. `python3 start_compag.py --text` remains available as a terminal fallback.

The application modules under `src/compag_curation`, their wheel, the native model ZIP, and the ten-photo ZIP are unchanged from rc9 R1–R3. The optional external YOLO segmentation checkpoint remains user-supplied and outside this public bundle; its scientific route and decision policy are unchanged. Software checks do not establish biological accuracy or independent human review.

This is a local delivery candidate. These notes do not claim that a GitHub repository or Release is live.
