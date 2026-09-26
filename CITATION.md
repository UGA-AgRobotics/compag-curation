> Current rc9 R4 notice (2026-09-25): The exact native r92 model ZIP and ten-photo ZIP are included in `bundled-assets/` under the author's selected CC BY 4.0 terms, recorded in `ASSET_LICENSES/PUBLIC_ASSET_SCOPE_EN.md`. No live release or rights-holder approval is attested. The JR09 paragraph below is historical.

> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Citation

## Software

Please cite the version of the software you used:

> Jahanifar, Hasan; Mirzakhaninafchi, Hasan; Porter, Wesley M.; Kiobia, Denis O.; and Rains, Glen C. (2026). Human-in-the-loop instance dataset curation for sticky-trap pest imagery using segmentation proposals and active learning (Version 1.9.3) [Computer software].

`CITATION.cff` contains the same software-release metadata in a
machine-readable form.

The cited software is a notebook-independent reference implementation with a
review-gated execution path for compatible sticky-trap imagery. Version 1.9.3
retains the sequential one-image active-learning workflow introduced in v1.8.1:
one image is proposed,
feature-extracted, uncertainty-reviewed, and followed by a fresh cumulative
XGBoost retrain into a lineage-bound `project-rN` bundle.

The public package does not contain the original study inputs, historical
training corpus, or complete historical model lineage and does not claim exact
reproduction of the paper results. The bundled r92 scoring closure is initial
review transfer assistance only; without the genuine r1-r92 corpus and
decision lineage, `project-r1` must not be cited or described as `r93`.

## Associated manuscript

The associated manuscript is identified as:

> Hasan Jahanifar, Hasan Mirzakhaninafchi, Wesley M. Porter, Denis O. Kiobia, and Glen C. Rains. "Human-in-the-loop instance dataset curation for sticky-trap pest imagery using segmentation proposals and active learning."

`COMPAG-D-26-02120` is the manuscript identifier. It is not a DOI and does
not imply publication or acceptance. A preferred article citation is
intentionally omitted until complete, authoritative publication metadata is
available.

Citation requests are scholarly requests and are not additional license
conditions.
