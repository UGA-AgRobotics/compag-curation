# COMPAG r92 rc4 — precise limitations

- The public round uses the fixed, separately supplied native r92 model. It scores every prepared tile, but does not train or update the model. Advanced continuation/next-round fitting is NOT INCLUDED. No new model identity or accuracy claim is implied.
- Image preparation targets the canonical yellow-card warped region. All of that region's far-edge-aligned 512×512 tiles are included. Pixels outside a detected card warp are outside this analysis region; an identity fallback uses the original frame.
- The reviewed export contains tile-coordinate COCO polygons and source/warp/offset metadata. It does not automatically convert newly reviewed polygons to original-photo coordinates.
- Machine SAM2 contours are proposals. The sample photographs have no supplied human labels. Any scripted review used in acceptance is software-test evidence only.
- The current image-to-feature route is the named canonical Full producer with SAM2.1 Hiera Large, ResNet50 IMAGENET1K_V2 and the recorded r92 PCA/prototype. It is not a claim of bitwise parity with the historical Only_codes reference or of biological accuracy.
- The complete image route requires the pinned GPU environment, official Full-profile assets and a CUDA GPU; the CPU-only environment supports model verification and tabular scoring/review/export, but cannot run SAM2/ResNet inference.
- `init-images` accepts 1–20 JPG/PNG images at least 512×512 with unique portable stems. Each output path must be new. A failed `infer-full` run does not claim partial coverage and must be rerun into a new output path after correcting the cause.
- The model and two example photos are a limited proposed public subset. Their exact asset-specific release terms and the GitHub destination remain to be recorded. No GitHub tag, release or DOI is part of this candidate.
