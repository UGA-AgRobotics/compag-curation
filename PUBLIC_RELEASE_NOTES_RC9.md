# COMPAG Curation 1.9.4rc9 — external CJ segmentation YOLO adapter

Rc9 integrates a separately supplied, one-class CJ YOLO instance-segmentation checkpoint for local inference. The package manifest, checkpoint SHA-256, Ultralytics 8.4.26 runtime and exact tile inference configuration are verified. A separate optional environment predicts every prepared tile, and a versioned box receipt binds the model, configuration, source tiles and box rows before the main science-GPU stage accepts it.

The adapter uses YOLO boxes and confidences for the existing SAM2 prompt and hybrid XGBoost/YOLO fusion modes (`prompt`, `fusion`, `both`). It verifies box-to-mask count agreement from the segment model but does not insert YOLO's instance masks into the canonical proposal pool. Raw XGBoost probability and effective fused decision remain separately recorded. Existing `off` mode and the older detection-only training route are retained. An external checkpoint can be explicitly attached to a verified project model set without calling it locally trained.

The supplied YOLO checkpoint is not in the public source, wheel, sdist or Release assets. The native r92 model and ten-photo ZIP remain byte-identical to rc8. No scientific accuracy or rights-holder approval for public redistribution of the external checkpoint is claimed. No remote publication is performed in this local task.
