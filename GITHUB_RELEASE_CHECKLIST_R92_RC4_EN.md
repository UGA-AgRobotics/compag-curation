> Historical rc4 two-photo checklist. Use README.md and docs/RELEASE_ASSETS_PLAN.md for the current rc6 R3 ten-photo candidate.

# GitHub publication checklist — rc4 local candidate

- [ ] Review the final positive allowlists and `SHA256SUMS.txt`; match source ZIP, wheel, sdist, unchanged native r92 ZIP and two-photo input ZIP to the output identity manifest.
- [ ] Record exact model and two-photo asset terms and final repository/release destination. Keep all other study images, labels, feature tables, private reviews and audit files outside the public release.
- [ ] Inspect the public source/wheel/sdist and sample ZIP for unintended files and verify attribution/notices.
- [ ] Confirm clean CPU installation and pinned GPU installation on the intended Linux/CUDA machine; keep their receipts. Run `r92 verify`, `init-images`, `infer-full` on every tile, human `review`, `export` and `verify-round` using fresh output paths.
- [ ] Distinguish software-test review labels from human decisions and do not cite sample cards as a held-out accuracy check.
- [ ] Publish only after final payloads, terms, repository permissions and release text are approved. No publication action is performed by this candidate task.
