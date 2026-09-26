# COMPAG Curation 1.9.4rc9 — reviewer zoom continuity (R5.2)

R5.2 keeps the current full-card review image at the same zoom and pan after Accept, Flip, uncertain decisions, Skip, Delete, Undo, and bulk acceptance refreshes on the same card. Selecting a different card still loads its image and fits it to the review canvas. This helps inspect small insects without repeated zooming between individual decisions.

Only the project reviewer display code changes in the installed application. Scored candidates, human decisions, saved review events, model predictions, inference, XGBoost/YOLO training, and native r92 model assets are unchanged. The application wheel is repackaged from R5.1 with only the reviewer module and wheel integrity record updated; it has a new pinned checksum despite retaining version 1.9.4rc9. Existing installations of the same version require a one-time force reinstall of the bundled wheel to receive this display fix. No review workspace data is migrated or rewritten.

For an existing default GPU installation, first stop the current reviewer and local guide. From the R5.2 repository folder, run:

```bash
SOURCE="$(pwd)"
PREFIX="$HOME/.local/share/compag-r92-rc9-gpu"
( umask 022; env -u PYTHONPATH -u PYTHONHOME "$PREFIX/bin/python" -I -B -m pip install --no-index --no-deps --no-compile --force-reinstall \
    "$SOURCE/bundled-assets/compag_curation-1.9.4rc9-py3-none-any.whl" )
env -u PYTHONPATH -u PYTHONHOME "$PREFIX/bin/python" -I -B -m compag_curation doctor --profile science-gpu
python3 start_compag.py
```

Use the installation folder chosen in the guide if it differs from the default `PREFIX`. The `umask 022` applies only to the reinstall subshell and keeps the installed package files at the permissions expected by the GPU profile. Restart the reviewer from the guide after it opens; use the newly printed local address.

The Python 3.10 launcher repair from R5.1 remains included. This is a local delivery candidate; no remote GitHub publication is claimed.
