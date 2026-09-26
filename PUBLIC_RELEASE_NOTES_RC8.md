# COMPAG Curation 1.9.4rc8 — reviewer image pan repair

The full-pool project review canvas now suppresses the browser's native context menu. Pressing and dragging with the right mouse button moves the image immediately, including on the first gesture. Left-button drag still pans; left click without a drag still selects a candidate. This applies only to the review image canvas, so normal browser menus elsewhere remain available.

This version reuses the rc7 inference implementation and its checkpoint code identity. Existing complete scored output, exact model, ten photographs, and review state can be reopened in rc8. No inference, classifier, threshold, mask, candidate or training decision code changed. The native model ZIP and ten-photo ZIP remain byte-identical to rc7. No rerun is required solely for this UI change.

The local GitHub candidate has been staged with a matching source ZIP, wheel, sdist and release assets. No remote publication or independent scientific validation is asserted.
