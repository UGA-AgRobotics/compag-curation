# Security

## Reporting

Use the repository host's private security-advisory feature when it is enabled.
If it is unavailable, open a sanitized public issue that requests a private
reporting route without describing the vulnerability. Do not post exploit
details, credentials, private research material, or local filesystem paths.

Once a private route is established, include the affected version/component, a
synthetic minimal reproduction, expected and observed behavior, impact, and
suggested mitigation.

## Boundary

The package is a local CLI with a local browser guide. Import and validation
require no credential. Asset fetches use the pinned registry. Treat user files
and bundles as untrusted; path checks and checksums are integrity checks.

- **Native r92 / public XGB:** verified JSON or UBJ classifier data and
  non-object NumPy transforms. The public classifier loader rejects pickle-like
  input, unsafe links/path swaps and unsupported extensions, and passes a verified
  byte buffer to XGBoost. This is the constrained native scoring route.
- **Optional YOLO:** `r92_yolo.py` and the segmentation helper hand trusted `.pt`
  checkpoints to the upstream Ultralytics/Torch loader. This route is not claimed
  pickle-free. Use only checkpoints from a trusted source; a user-supplied hash
  or manifest does not authenticate the publisher or establish safe deserialization.
- **Explicit-trust legacy conversion:** `only_codes/model_bundle.py` retains a
  one-time legacy pickle conversion requiring explicit trust and a matching SHA-256.
  Pickle can execute code. Do not convert untrusted input. Converted output uses
  the constrained format; the conversion itself remains a sensitive trust boundary.
- **Legacy data packs:** `only_codes/safe_joblib.py` uses a restricted NumPy
  unpickler with an allowlist. This is a separately bounded legacy-data reader,
  not a general model loader or permission to accept arbitrary pickle objects.

`joblib` also remains in the locked science environment for computation.
No loading permission is broadened by this release. Linux Stop owns only the
session/process group it creates and retains the unreaped leader until cleanup
is confirmed. Workers must remain in that group. Cleanup failure blocks new
work; it is not reported as a successful Stop.

## Authenticity And Isolation

A bundle or output self-manifest detects changes relative to the manifest that
was supplied with it; it is not a digital signature and does not authenticate a
publisher or release. Before trusting a release artifact, obtain its expected
hash through an independently delivered, trusted publisher channel and compare
it out of band with the artifact you received.

Manifest and hash verification do not make a truly untrusted model or image
safe from defects in native decoders or model runtimes. Inspect such material in
an isolated, least-privilege OS account, sandbox, or disposable virtual machine
without credentials or network access, and copy out only reviewed results.

This document describes the 1.9.4rc10 / R5.4 source line. Third-party advisories and
licenses remain the user's or institution's responsibility.
