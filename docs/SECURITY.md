# Human-Review Application Security

This document covers the v1.7 `review-ui` web, desktop, and status surfaces.
Use the repository-root `SECURITY.md` for vulnerability reporting.

## Trust Boundary

The reviewer accepts one of two immutable source sets. Initial review validates
the paused-run owner and lock identities, event ledger, exact Stage 0/10/20
receipts, pause event, review-request hash, proposal table, overlay manifest,
and served overlay hashes. It holds a shared run lock so scientific resume
cannot race an open initial-review session. Round review validates the whole
sealed `begin_round` operation root, its exact bound Stage-60 directory record,
predictions/shortlist/request and bundle/profile lineage, and the complete
original inference-image inventory. It rejects a copied selection subdirectory,
a different Stage 60, changed images, or output/state paths inside any immutable
input. Project-1/CJ assistance additionally verifies the exact bundled
UBJ/JSON/NPY r92 closure and binds the score/order cache to the immutable
Stage-20 evidence. The reviewer never writes inside an input and never modifies
`review_request.csv`.

The final reviewed CSV and state directory are private user data. Keep both
outside the repository, do not synchronize them to a public location, and do
not share them as release evidence. State is created private to the current
user, uses a single-writer lock, and stores every decision/undo as a
hash-chained append-only event. The final CSV is validated and published with
no-clobber semantics. Symlinks, path traversal, unsafe ownership/modes,
overlapping input/state/output paths, changed bindings, and an existing output
are rejected.

## Web Mode

Web mode is a local application, not a network service. It can bind only to
`127.0.0.1`; remote or wildcard hosts are rejected. A random high-entropy
capability token is placed in the private startup URL, API requests must supply
it, mutating requests must have the exact loopback origin, and the Host header
must match the selected local port. The browser removes the token from the
visible address after loading, but the startup URL remains a secret: do not
paste it into chat, logs, screenshots, issue reports, or another machine.

Responses disable caching, framing, referrers, MIME sniffing, and cross-origin
resource access. A restrictive nonce-based Content Security Policy allows only
the embedded application and same-origin API/images. Request schemas and body
sizes are bounded. Initial review serves only hash-verified PNG overlays from
the sealed manifest. Round review re-hashes the bound original image before
rendering a bounded PNG crop with its Stage-60 polygon and bounding box. User
paths are not exposed as arbitrary HTTP routes.

Use the default random port. `--port N` is for a required local fixed port, not
for remote access. `--no-open-browser` suppresses automatic browser launch but
does not weaken authorization. Close the UI from its button or stop the local
process when finished. Do not reverse-proxy, port-forward, container-publish,
or rebind this interface.

## Desktop And Status Modes

Desktop mode uses local Tk/Pillow and the same session engine; it does not open
a listener. Run it only in a trusted Ubuntu desktop, WSLg, or explicitly
trusted X11 session. A remote X server can observe UI content and input and is
outside the accepted release profile.

Status mode opens the same bound state and prints progress JSON. If the state
does not yet exist, it creates the private session directory, so treat it as a
stateful local inspection. For Project-1/CJ, first creation can perform the
same verified Full-only CUDA r92 scoring needed to bind assistance; later opens
reuse the verified cache. It does not write inside immutable inputs, finalize
the CSV, resume a pipeline/round, or train.

The installed Ubuntu `.desktop` launchers use terminal-backed generated shell
launchers with literal interpreter/project paths, normalized absolute path
checks, no `eval`, and no-clobber output rules. The round launcher requires the
selection, Stage 60, and image directory separately and states that it only
records review. It does not invoke `active-learning resume-round` or training.

## Human-Control Guarantees

No mode silently converts a prediction into a human decision. Auto-next only
changes the selected proposal after a decision has been durably recorded.
Cold start and project-trained rounds require explicit proposal-by-proposal
choices. Project-1/CJ may display r92 predictions and offers one separately
labeled, explicitly confirmed, reversible remainder operation; confirmed rows
use uncertain actions at weight 0.4 and distinct provenance, while manual
corrections remain weight 1.0. Skip is saved only after explicit selection as
`label=0`, `review_action=skip`, and `review_weight=0.0`. Finalization requires
no pending rows and a second explicit confirmation. Closing a UI never
finalizes and no UI invokes scientific resume or training.

## Version Compatibility

Keep release environments separate. v1.7 may inspect and finalize a compatible
paused v1.4 or v1.5 review request, but the run's package/dependency identity remains
v1.4. Resume with the unchanged v1.4 environment that created that run. Do not
upgrade it in place, edit run receipts, or bypass a resume-drift failure.
