# COMPAG Curation 1.9.4rc9 — easy-start source revision R2

This source revision adds `start_compag.py` and a desktop guide with a terminal fallback. After a GitHub clone, the guide checks the four required files from the matching Release Assets against fixed SHA-256 identities (and checks the Release manifest when supplied), runs the existing pinned GPU installer and doctor, prepares the ten example cards, downloads/verifies official SAM2 and ResNet assets, and offers full-card analysis, review, interruption resume, explicit XGBoost training and next-card model selection. User projects are kept outside the source repository.

The application modules under `src/compag_curation`, the wheel `compag_curation-1.9.4rc9-py3-none-any.whl`, the native r92 model ZIP, and the ten-photo ZIP are byte-identical to rc9 R1. Only onboarding code, documentation, source ZIP and source distribution have changed. The external YOLO segmentation checkpoint remains optional and outside the public Release; its scientific route and decision policy are unchanged.

This local candidate has not been uploaded to GitHub. The software checks do not establish biological accuracy or independent human review.
