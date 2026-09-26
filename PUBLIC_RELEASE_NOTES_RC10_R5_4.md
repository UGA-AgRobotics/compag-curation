# R5.4 / COMPAG Curation 1.9.4rc10

Parent: audited R5.3 source ZIP SHA-256
`f7c3c5eb79021dda5c2937302c7598f06e81a5660a1471bc00bed3c20a623b2a`.
This revision advances the application identity and rebuilds version-bound
runtime metadata and installed documentation. Dependency versions, scientific
numerical routines, paper-training settings and the two selected asset ZIPs
remain unchanged.

Stop now owns the process group until all live members have ended, including
children whose captured pipes have closed. Bounded escalation and explicit
cleanup completion keep Stop pending and block a conflicting job. Repeated
Stop is idempotent; stale browser job IDs cannot cancel a later job. An
unconfirmed cleanup is reported as failure and keeps conflicting work blocked.
Completed artifacts and review history are retained. Training restarts fresh;
only the existing completed-tile inference cache supports continuation.

COCO export preflights the entire session list. Duplicate resolved paths and
aliases are rejected before output; EASY_SESSION v1's absolute `path` is its
saved run identity, so a copied record retaining that identity is the same
session. Legacy records lacking this field use their resolved directory.
Distinct genuine runs keep distinct path identities and remain separate
annotation sets even when photos or masks match. Current and reviewed scopes,
mask coordinates, provenance, and no-overwrite publication remain unchanged.

The quick-demo output **basename** must be printable ASCII (spaces allowed).
This is not a blanket restriction on Unicode workspace/environment paths.

Current documentation distinguishes native safe-format scoring, optional
upstream YOLO checkpoint loading, explicit-trust pickle conversion and the
restricted legacy data reader. See the controlled-input replay runbook for
paper-result scope; no global historical reproduction or publication is claimed.

Run the documented base suite in `docs/DEVELOPMENT.md`. With the installed
CPU or science COMPAG Python, also run:

```sh
python -I -B tools/test_easy_start_workflow.py
python -I -B tools/test_release_repairs.py
```

The lifecycle regression includes 20 synchronized closed-stdio descendant
cancellations. These are software tests, not genuine human/biological validation.

Upgrading: use a new installation folder and new workspace for 1.9.4rc10. Existing workspaces are preserved. To export earlier saved sessions without changing their settings, use `tools/export_session_coco.py --session /path/to/session --scope current --output /path/to/new.zip` with the installed COMPAG Python. Training a stopped operation begins a new fit.

The optional segmentation environment helper preserves a venv Python path and installs the verified matching release wheel in a newly created child when nested-venv inheritance would otherwise expose the older application. Existing environments are never overwritten. The pinned provider dependencies remain unchanged.

`CONTRACT_FIX_SOURCE_TO_DERIVATIVE_MAP.json` retains its dated 20260924 historical mapping and is not a current tree checksum manifest. `PUBLIC_SOURCE_ALLOWLIST.json` defines current membership; the delivered ZIP sidecar binds the actual release bytes.
