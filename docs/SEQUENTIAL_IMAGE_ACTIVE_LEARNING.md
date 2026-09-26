> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Strict Post-r92 One-Image Active Learning

## Scientific Scope

COMPAG Curation 1.9.3 retains the strict transfer workflow introduced in
v1.8.1, in which one
operational active-learning round corresponds to exactly one previously unseen
image and one fresh XGBoost fit:

1. generate SAM2 proposals and canonical four-scale features for one image;
2. score that image with the published r92 classifier in round 1, or with the
   immediately preceding `project-rN` bundle in later rounds;
3. select only the deterministic Top-K proposals (`K=50`) inside the fixed
   uncertainty margin, meaning 1 through 50 rows when fewer than 50 qualify;
4. collect an explicit human action for every selected proposal; and
5. fit XGBoost from scratch over the verified transfer baseline plus all
   cumulative reviewed image-round rows.

The fixed acquisition policy is XGBoost-only: threshold `0.5`, uncertainty
distance `abs(xgb_p - 0.5)`, inclusive eligibility margin
`0.3 <= xgb_p <= 0.7`, at most 50 proposals, then `proposal_id` as the
deterministic tie break. If no proposal is eligible, the operation fails closed
without publishing a review batch or a model.

One image per round is a stricter operational schedule around the paper's
proposal-batch active-learning mechanics. It is not a claim that the historical
study defined each round as one image or followed the same image order.

## Private Transfer Baseline

Round 1 does not require an exhaustive new genesis review. Instead, the user
must separately supply the exact recovered r92 fit-time feature snapshot:

```text
SHA-256: 1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d
size:    582968556 bytes
```

That private CSV is never included in the public repository, source ZIP,
wheel, sdist, online installer, public model resources, or GitHub Release. The
public command accepts no substitute: both the SHA-256 and size must match
before it creates a no-clobber, hash-sealed local baseline.

The import retains all 90,887 complete, labeled, non-skip proposals (363,548
four-scale rows), including the historical distinction between reviewed and
unreviewed rows. Unreviewed baseline rows retain blank review-action provenance
and unit numerical weight; a trainer adapter may present their existing labels
to XGBoost, but it does not rewrite them as human `accept` decisions.

The importer creates a deterministic group-pure train/test split and verifies
class-valid group folds. That split is a transfer-evaluation split, not a claim
that the unavailable historical r92 split was reconstructed.

## Frozen Published State And Claim Boundary

The bundled public `compag-cj-r92` resource supplies only the safe portable
r92 classifier and its frozen published prototype/PCA state. It does not
supply the private transfer baseline or the complete historical r1-r92 review
and retraining trajectory.

Every transfer bundle records:

```text
workflow:           SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1
lineage:            POST_R92_REVIEWED_TRANSFER_BASELINE
reproduction:       NOT_R92_REPRODUCTION
frozen state:       FROZEN_PUBLISHED_R92_PCA_AND_PROTOTYPE
feature scope:      FROZEN_PUBLISHED_R92_TRANSFER_STATE
CV interpretation: CONDITIONAL_ON_FROZEN_R92_TRANSFER_PROTOTYPE_AND_PCA
```

PCA explained variance reported for a transfer fit describes the observed
variance of the transfer-training projections under the frozen r92 PCA. It is
not represented as r92 fit-time PCA variance.

The first newly trained bundle is therefore `project-r1`, never `r93`. Later
successful rounds publish `project-r2`, `project-r3`, and so on.

## Round 1

Every path below must be absolute or resolve to a safe, new path. The image
directory must contain exactly one supported image. Keep private data and all
generated outputs outside the source repository.

```bash
set -euo pipefail

VENV="$HOME/.local/share/compag-curation/venvs/science-gpu-full-1.9.3"
PY="$VENV/bin/python"
PROJECT="$HOME/compag-workspaces/compag-full-project-v183"
CONFIG="$PROJECT/config.toml"
PRIVATE_R92_FEATURES="$HOME/private-compag-transfer/features_train.csv"
BASELINE="$HOME/compag-workspaces/compag-r92-transfer-baseline"
AL="$HOME/compag-workspaces/compag-post-r92-al"

IMAGE_DIR="$PROJECT/data/image-round-001"
INFERENCE="$AL/round-001-inference"
SELECTION="$AL/round-001-selection"
REVIEWED="$PROJECT/review/image-round-001-reviewed.csv"
COMPLETION="$AL/round-001-completion"

mkdir -p "$AL"

"$PY" -I -B -m compag_curation active-learning import-r92-transfer-baseline \
  --source-features "$PRIVATE_R92_FEATURES" \
  --output "$BASELINE"

"$PY" -I -B -m compag_curation active-learning infer-r92-image \
  --config "$CONFIG" \
  --images "$IMAGE_DIR" \
  --output "$INFERENCE"

"$PY" -I -B -m compag_curation active-learning begin-transfer-image-round \
  --stage60-root "$INFERENCE" \
  --transfer-baseline "$BASELINE" \
  --round-number 1 \
  --output "$SELECTION" || test "$?" -eq 3

"$PY" -I -B -m compag_curation review-ui web \
  --selection-root "$SELECTION" \
  --stage60-root "$INFERENCE" \
  --images "$IMAGE_DIR" \
  --output "$REVIEWED"

"$PY" -I -B -m compag_curation active-learning resume-transfer-image-round \
  --selection-root "$SELECTION" \
  --stage60-root "$INFERENCE" \
  --transfer-baseline "$BASELINE" \
  --config "$CONFIG" \
  --reviewed "$REVIEWED" \
  --round-number 1 \
  --output "$COMPLETION"
```

Closing the reviewer does not finalize a review. Run the resume command only
after the UI has explicitly finalized the complete reviewed CSV. The reviewer
does not infer, resume, or train automatically.

Round 1 deliberately rejects `--source-bundle`, `--decision-log`, and
`--prior-completion`: the scorer must be the verified published r92 resource.
Its completed bundle is:

```text
round-001-completion/payload/round_completion/model_bundle
```

and its provenance label is `project-r1`.

## Round 2 And Later

For round `N > 1`, the scorer must be the bundle created by round `N-1`. The
command also requires the prior completion's exact decision log and the whole
prior `resume-transfer-image-round` operation root. Loose, copied, or skipped
ancestry is rejected.

For example, round 2 is:

```bash
set -euo pipefail

PRIOR="$AL/round-001-completion"
PARENT_BUNDLE="$PRIOR/payload/round_completion/model_bundle"
PRIOR_DECISIONS="$PRIOR/payload/round_completion/decision_log.csv"
IMAGE_DIR_2="$PROJECT/data/image-round-002"
INFERENCE_2="$AL/round-002-inference"
SELECTION_2="$AL/round-002-selection"
REVIEWED_2="$PROJECT/review/image-round-002-reviewed.csv"
COMPLETION_2="$AL/round-002-completion"

"$PY" -I -B -m compag_curation infer --execute \
  --images "$IMAGE_DIR_2" \
  --bundle "$PARENT_BUNDLE" \
  --output "$INFERENCE_2" \
  --device cuda

"$PY" -I -B -m compag_curation active-learning begin-transfer-image-round \
  --stage60-root "$INFERENCE_2" \
  --transfer-baseline "$BASELINE" \
  --round-number 2 \
  --source-bundle "$PARENT_BUNDLE" \
  --decision-log "$PRIOR_DECISIONS" \
  --prior-completion "$PRIOR" \
  --output "$SELECTION_2" || test "$?" -eq 3

"$PY" -I -B -m compag_curation review-ui web \
  --selection-root "$SELECTION_2" \
  --stage60-root "$INFERENCE_2" \
  --images "$IMAGE_DIR_2" \
  --output "$REVIEWED_2"

"$PY" -I -B -m compag_curation active-learning resume-transfer-image-round \
  --selection-root "$SELECTION_2" \
  --stage60-root "$INFERENCE_2" \
  --transfer-baseline "$BASELINE" \
  --config "$CONFIG" \
  --reviewed "$REVIEWED_2" \
  --round-number 2 \
  --source-bundle "$PARENT_BUNDLE" \
  --decision-log "$PRIOR_DECISIONS" \
  --prior-completion "$PRIOR" \
  --output "$COMPLETION_2"
```

Repeat the same pattern for each new image, advancing the round number and
using only the immediately preceding completion, decision log, and bundle.

## Fresh Full Retraining

`resume-transfer-image-round` never appends trees to the parent classifier. It
starts a fresh XGBoost fit over:

```text
verified private post-r92 transfer-baseline rows
+ all non-skip reviewed raw feature rows from prior one-image rounds
+ all non-skip reviewed raw feature rows from the current one-image round
```

The transfer baseline and frozen r92 feature state are verified again at the
resume boundary. Confident human actions carry weight `1.0`, uncertain actions
carry weight `0.4`, and skip carries weight `0.0`. Only the selected Top-K
(`K=50`, or every eligible row when fewer qualify) proposals from a new image
may receive round decisions or enter the cumulative review archive. Unselected
proposals are never silently converted into labels.

## Fail-Closed Invariants

A transfer image-round is rejected before publication or training if any of
these conditions is violated:

- the private source is not the exact required SHA-256 and byte size;
- the public r92 classifier, feature order, prototype, PCA, or resource
  manifest changed;
- the inference root contains zero images or more than one image;
- the image/group was already present in the baseline or a prior round;
- identical image bytes reappear under another name or group;
- round 1 was not scored by the published r92 resource;
- a later round was not scored by the immediately preceding `project-rN`;
- the baseline, split, decision log, source bundle, or completion ancestry
  changed or is discontinuous;
- the reviewed CSV is incomplete, belongs to another selection, or changed;
- a selected proposal's raw feature rows are missing or changed;
- the new bundle fails its lineage, CUDA/XGBoost, or UBJ parity checks.

All operations are no-clobber. Preserve failed and superseded runs as evidence;
do not merge their review state or artifacts into this lineage.

## Separate Older Workflows

The general `begin-round` / `resume-round` pool workflow and the v1.8.0
genesis-based `begin-image-round` / `resume-image-round` contract remain
separate interfaces. They must not be mixed with the strict post-r92 transfer
commands, baseline, decisions, or ancestry described here.
