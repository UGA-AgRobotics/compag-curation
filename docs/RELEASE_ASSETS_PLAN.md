# Public bundled assets — rc9 easy-start R4

The public GitHub repository includes four regular files in `bundled-assets/`. They are part of an ordinary clone and of the reviewed source ZIP. The browser guide checks their pinned identities automatically; the user does not download a separate Release asset or enter a path.

| Included file | SHA-256 | Purpose |
| --- | --- | --- |
| `compag_curation-1.9.4rc10-py3-none-any.whl` | `3d3f76e94cfabfef616fd1b2809f94800a50da0ee601124feb7709dc653a32cb` | Matching application wheel |
| `compag_curation-1.9.4rc10-py3-none-any.whl.sha256` | Text sidecar containing the wheel hash and name | Wheel identity check |
| `COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip` | `fffef7a85192b99dcca3bcc5750215150d2ca9659121363198041244bbfe57c1` | Immutable native r92 model |
| `COMPAG_R92_TEN_PHOTO_INPUTS_RC6_R3_20260924.zip` | `344c8d767a1719403a3713e0f51587971f4ab9ea18e1d1a44034ed72c9297914` | Ten author-selected original full-card JPEGs |

The photo archive retains its rc6 R3 filename because its bytes are unchanged. The model and photographs use author-selected CC BY 4.0 terms, subject to rights-holder authority verification. No other research photos, annotations, real review logs, or newly fitted project weights are bundled. The external YOLO segmentation checkpoint remains optional and user-supplied; see `EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md`.

The local delivery may also include a separate `release-assets/` mirror and `PUBLIC_SHA256SUMS` for publication and provenance. These are not a prerequisite for a user who cloned the repository. No live GitHub repository or Release is claimed by this local candidate.
