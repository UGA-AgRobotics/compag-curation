# Development and local release validation — rc9

This source tree is packaging revision `rc10-repair-r5.4` of application
`1.9.4rc10`. Run the following bounded command from a fresh extraction of the
public repository ZIP with a reviewed CPython 3.12 environment containing the
`setuptools==84.0.0` and `wheel==0.48.0` build backend declared in
`pyproject.toml`. Choose new paths outside the source tree:

```bash
python3.12 -I -B tools/public_safe_acceptance.py \
  --root . \
  --work-dir /absolute/absent/compag-rc10-r5-4-acceptance-work \
  --receipt /absolute/existing-parent/compag-rc10-r5-4-acceptance.json
```

The runner rejects existing output paths and unexpected revision/application
identities. It copies this tree, creates a fresh local environment, installs
the source without dependency resolution or network access, checks the base
CLI/doctor, and executes exactly the active synthetic tests listed in
`PUBLIC_TEST_INVENTORY.json`. All must pass without skips. Historical, deferred,
and restricted tests receive no pass credit. This validation does not load the
native r92 model or certify human-reviewed science. The guarded paper-fit helper
and optional external YOLO path need their own science-environment and model
checks; they receive no pass credit from this model-free lane.

For the complete example assets, use the four files included in `bundled-assets/`
as described in `README.md`. R5.3 changes the optional YOLO-enabled hybrid
decision and review score to the paper formulation and updates the browser's
mode names. The canonical YOLO-disabled route and R5.2 same-card review zoom
behavior remain unchanged. Focused synthetic hybrid regressions are in
`tests/test_r92_round_continuation.py`; the model-free acceptance command above
does not execute YOLO inference. The mask display fallback introduced before
rc9 differs from rc6 R3; native model assets remain unchanged.
Keep generated user projects and test outputs outside this repository tree.

The browser launcher is also checked against system Python 3.10. With an existing
workspace containing a trained XGBoost model, the local status endpoint must
return verified training evidence instead of dropping the connection.

R5.4 adds lifecycle and COCO session-identity regressions in `tools/test_release_repairs.py`, including 20 controlled cancellations. Run it with the installed COMPAG Python alongside `tools/test_easy_start_workflow.py`. These supplemental checks are separate from the 16-case bounded base inventory above.
