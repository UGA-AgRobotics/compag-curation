# Saved COCO exports, stop controls, and paper-training readiness

This R5.3 launcher update keeps the scientific application wheel, models,
inference settings, and the existing paper XGBoost trainer unchanged.
Restart `python3 start_compag.py` from this directory; reinstalling is unnecessary.

## Export saved masks

The **Export saved masks** panel is available below every workflow step.
Choose one completed analysis or **All completed analyses**, choose the labels,
then click **Export COCO ZIP**. Close an open review using **Finish review and
continue** first so its decisions cannot change during export.

* **Current labels + model predictions** includes saved class labels and the
  current inference's predictions for candidates without a review decision.
* **Saved labels only** includes individual and bulk-accepted labels. It can
  produce zero annotations if no labels have been saved yet.
* Deleted and skipped candidates are excluded in both modes. Both CJ and non-CJ
  classes are included. Each annotation records its label origin and review action.

Each ZIP contains `annotations.json`, the original images under `images/`, and
`EXPORT_RECEIPT.json`. Binary masks are transformed back to original-photo pixels
and encoded as standard uncompressed COCO RLE, retaining holes and disconnected
parts. Display bounding boxes are never substituted for segmentation masks.
Overlapping candidates are preserved without merging. Exporting all analyses
also preserves separate runs of the same photo; they can contribute multiple
annotation sets to the same original image.

Exports are saved under `coco-exports/` in the private workspace. Download links
remain available after restarting the guide (the 30 most recent exports are
listed; older files remain in that folder). Repeated exports use new filenames.
Review completion, finalizing a training snapshot, and fitting a model are not
required. An interrupted analysis must be resumed to completion before export;
an incomplete photo is never presented as a complete dataset.

## Stop analysis or training

While a job runs, the progress panel shows **Stop analysis** or **Stop training**
(and **Stop export** during export). The button first interrupts the owned process
group, including detector/trainer child processes. If a native operation does not
respond, the group is terminated and then killed after bounded waits. The guide
shows **Stopping** until the worker finishes and blocks conflicting jobs.

Saved review decisions and completed inference checkpoints remain available.
Use **Continue analysis** to resume with completed tiles. Stopped training starts
again as a fresh attempt; this does not promise resuming a partially fitted
XGBoost search. A completed, verified model that finished before Stop is retained.
An interrupted output is never reported as a successful fit or complete export.
Closing the guide with Ctrl+C also cancels its active child process group.

## Paper protocol remains unchanged

**Check saved decisions** now explains each current blocker and counts independent
photos using original-photo hashes. When the basic label and photo checks pass,
it runs the existing trainer's full preflight on a temporary snapshot. Only a
successful preflight enables the Paper protocol training button. Training still
revalidates the data before fitting.

Section 2.8 and Table S1 of the supplied manuscript specify GroupKFold with five
splits, 30 sampled configurations, a group-safe 0.30 early-stopping validation
split, and 30 early-stopping rounds. They do **not** explicitly say “at least six
photos.” Six is a necessary lower bound in the existing implementation, which
additionally holds out separate original-photo groups for testing. The actual
split and both-class checks can require more photos. This update does not change
that algorithm, accept bulk predictions as individual reviews, or weaken the
validation splits.

To build eligible data, start a clean lineage with **Original model**, save
individual Accept/Flip decisions across photos containing both classes, and use
Operational training plus **Use latest trained model** to accumulate clean
rounds. Avoid **Accept remaining** in that lineage. Recheck Paper protocol after
collecting enough independent reviewed photos. An inherited bulk acceptance
remains in the cumulative data even if the current photo is individually reviewed.

## Validation

Focused tests cover stop/retry, unresponsive child processes, request protections,
COCO geometry/RLE, reviewed and unreviewed exports, deleted/skipped candidates,
tamper rejection, and paper preflight with clean/bulk/insufficient-group fixtures.
Frontend event tests check Stop during active work and training-button eligibility.
No full scientific retraining is performed for this launcher update.
