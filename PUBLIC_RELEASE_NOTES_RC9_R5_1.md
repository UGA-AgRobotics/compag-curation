# COMPAG Curation 1.9.4rc9 — Python 3.10 browser-start repair (R5.1)

R5.1 repairs the local browser guide when it starts with Python 3.10 and an existing workspace already contains a trained XGBoost model. The guide now uses its existing streaming SHA-256 helper to verify saved model evidence. R5 used `hashlib.file_digest`, which is unavailable in Python 3.10; its status request then closed without a response and the page displayed “Failed to fetch.”

The browser guide, installed scientific engine, application wheel, native r92 model ZIP, ten-photo ZIP, and training results retain their R5 behavior. No existing user workspace or review decisions are migrated or rewritten. Restart the local guide after replacing R5 with R5.1, and open the new local address printed by `python3 start_compag.py`.

The focused regression check creates a synthetic saved-model evidence tree, removes `hashlib.file_digest` from the test runtime, and requires verified evidence. A separate Python 3.10 status-endpoint smoke confirms an HTTP 200 response for an existing trained workspace. The broader model-free acceptance remains the bounded Python 3.12 lane documented in `docs/DEVELOPMENT.md`; neither check asserts scientific accuracy.

This remains a local delivery candidate. No remote GitHub publication is claimed. The R5 workflow and its limits are documented in `PUBLIC_RELEASE_NOTES_RC9_R5.md`.
