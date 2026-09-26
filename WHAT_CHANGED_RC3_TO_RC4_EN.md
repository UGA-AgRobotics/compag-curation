# What changed from rc3 to rc4

- `r92 init-images` now prepares and hashes **every** canonical tile for every selected image; rc3 prepared one selected tile per image.
- New `r92 infer-full` runs the existing Full-profile SAM2/ResNet producer and fixed r92 scorer across the complete prepared tile registry. Its receipt records each tile, including tiles with zero proposals, and refuses missing/modified inputs.
- Review state binds the full inference receipt. Export checks that binding and writes a hashed output manifest. New `r92 verify-round` checks full coverage, scores, export and original preservation.
- The public first-run path includes a separate pinned GPU installer, a two-photo input pack, explicit limitations and a GitHub release checklist.
- The native r92 model, 93-feature order, fitted transforms, CPU scorer and inherited scientific core are retained. No continuation fit or new accuracy claim was added.
