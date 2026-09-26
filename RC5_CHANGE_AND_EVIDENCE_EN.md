# rc5 local change and evidence report

## Starting point

The exact rc4 laptop/public candidate already supplied strict native r92 verification and scoring, all-tile canonical Full preparation/inference, a review/export bridge, and a verified 88/88-tile two-photo fixed-model run with 2,589 scored candidates. The native model archive, frozen Only_codes behavior, and historical manuscript results were retained. Rc4 did not connect a selected review batch to a user-owned cumulative training snapshot, fresh project XGBoost fit, optional completeness-certified YOLO fit, versioned model set, and next-card inference. `RC4_GAP_MAP_EN.md` gives the source and test map.

## Implemented in rc5

- A selected, resumable review batch with ordered effective decisions; skip, context, and duplicate accounting; a hash-bound cumulative snapshot with frozen native r92 feature state; a real fresh XGBoost fit, verifier, and next-card scorer.
- An original-coordinate COCO export of selected reviewed **full masks**. It does not assert exhaustive image annotation. Geometry changes require newly generated compatible features and a new identity.
- Explicit complete-tile external QA for one-class CJ detection, independent source-card splits, empty-target examples, an isolated real Ultralytics 8.3.221 fit, checkpoint reload/probe, and verified all-tile box precomputation.
- Independent detector OFF, SAM2 box PROMPT, hybrid matched-score FUSION, and BOTH routing. The XGBoost-only default does not import or require Ultralytics. A requested bad checkpoint fails explicitly.
- Verified round model sets with atomic latest-pointer activation, explicit older-model selection, durable stage state, no-clobber outputs, and hash-bound per-tile interruption checkpoints. A changed model, detector policy, boxes, project, assets, or implementation invalidates cached tiles.
- New CLI and interactive launcher choices, an installed-wheel acceptance runner, regression tests, and a complete-round runbook. The native r92 archive is separate and unchanged.

## Local execution evidence

These diagnostic photographs and fixture decisions are software-operation evidence, not independent biological validation.

| Capability | Evidence level | Actual result | Local evidence location |
| --- | --- | --- | --- |
| Complete first-card inference | Real full-image inference, native r92 | 48/48 tiles; 1,562 candidates; scores match the rc4 fixed-r92 run | `first_card_inference/FULL_INFERENCE_RECEIPT.json` |
| Actual human review preparation | Real review state, awaiting user | 50 raw-XGB uncertainty selections; state opens/reopens; zero human decisions | `first_card_selection.csv`, `first_card_review_pending/` |
| Review finalization and original masks | Software-test decisions on real diagnostic image | 51 ordered events for 50 selections; 48 eligible rows, 2 skips; 48 mapped original-coordinate full-mask COCO annotations | `software_test_r1/training_snapshot.json`, `software_test_r1/selected_original_coco.json` |
| Fresh project XGBoost fit | Real numerical fit on software-test labels | New 30-round XGBoost 2.1.1 classifier SHA-256 `462914dd339a9744f9a217424dd87bd5dc905731e9c93295be55cfebba0ae01b`; bundle freshly reloaded | `software_test_r1/xgb_model/`, `software_test_r1/ROUND_STATE.json` |
| Next card, YOLO OFF | Real full-image inference from installed wheel | 40/40 tiles; 1,027 candidates scored by the new project model; no Ultralytics installed in science-GPU environment | `second_card_off_inference/FULL_INFERENCE_RECEIPT.json`, `installed_acceptance_final3.json` |
| Detector QA guards | Synthetic contract fixtures | Four complete 512-pixel tiles from two source groups, two positive boxes and two certified empty units; incomplete QA, invalid box and cross-card split rejected | `yolo_fixture_v2/synthetic_yolo_dataset/DATASET_MANIFEST.json`, contract tests |
| Optional detector training | Real one-epoch CPU optimization on synthetic eligible tiles | Ultralytics 8.3.221 saved, freshly reloaded and predicted from checkpoint SHA-256 `3ed2e07f5df827c548887b970250a598f471d26190f00e41afa7ae10d5b2176f`; an additional installed-wheel fit with dependency/hardware/optimizer-policy metadata produced checkpoint SHA-256 `e75a05851e8d1c7846031e92754c6c281ebd399e380022d2457e7531800ed0f1`; zero validation probe detections in both; no accuracy claim | `yolo_fixture_v2/synthetic_yolo_fit_final3/YOLO_PROJECT_MODEL.json`, `yolo_fixture_v2/synthetic_yolo_fit_final5/YOLO_PROJECT_MODEL.json` |
| All-tile detector inference | Real prediction using the synthetic-fit checkpoint | 40/40 second-card tiles, 1,720 low-threshold boxes; box identity bound to tile and weight hashes | `second_card_yolo_boxes_final3.json` |
| Next card with XGBoost plus YOLO fusion | Installed-wheel real full-image inference | 40/40 tiles, 1,027 candidates, 49 matched detector supports; raw XGBoost scores, feature table, and proposal table byte-identical to the OFF route. The final distributable wheel independently repeated 40/40 with the same feature/proposal bytes and zero failed or excluded tiles. | `second_card_fusion_inference/FULL_INFERENCE_RECEIPT.json`, `second_card_fusion_final6_inference/FULL_INFERENCE_RECEIPT.json`, `installed_acceptance_final6.json` |
| Detector PROMPT route | Installed-wheel real SAM2 on a one-tile diagnostic crop | 1/1 tile; 43 real detector boxes; 15 surviving YOLO-prompted masks among 75 candidates; prior interrupt and failure reason retained in `PROGRESS.json` | `route_prompt_inference/FULL_INFERENCE_RECEIPT.json`, `route_prompt_checkpoint/PROGRESS.json` |
| Detector BOTH route | Installed-wheel real SAM2 and retained hybrid scoring on the same one-tile crop | 1/1 tile; same 75 candidate features and 15 prompted masks as PROMPT, with 14 matched detector supports and separate fused scores | `route_both_inference/FULL_INFERENCE_RECEIPT.json` |
| Optional failure isolation | Deliberately missing detector initialization | `YOLO_FAILED` recorded; existing fitted XGBoost classifier remained byte-identical | `yolo_fixture_v2/must_fail_missing_initial.YOLO_FAILED.json` |
| Interruption recovery | Real GPU inference interruption and completion | After three complete cached tiles, process interrupted; no partial PASS/output; resumed from a new output path and reused all three tiles in the 40/40 completed fusion run | `second_card_fusion_checkpoint/`, `second_card_fusion_interrupted.log`, `second_card_fusion_inference/FULL_INFERENCE_RECEIPT.json` |
| Existing compatibility and guards | Installed-wheel tests | Five new continuation tests pass; 23 Only_codes compatibility tests pass with 13 source-delegated skips; seven additional v1.9.1 fix tests pass; four review bridge negative guards pass | `tests/test_r92_round_continuation.py`, `tools/test_r92_bridge_guards.py` |

The incomplete 50-selection review was rejected at finalization. A changed feature/mask table was rejected. Re-finalizing identical candidates against the parent snapshot reported 48 repeated candidates and retained 48 cumulative rows. Failed activation with a missing YOLO checkpoint left the latest pointer byte-identical. A retained detector was verified against its exact earlier model set; a wrong source round was rejected. Changed YOLO mode invalidated the tile checkpoint. These are local software checks; they do not convert scripted labels into human labels.

The inherited `tests/test_review_launcher.py` still imports the absent historical `tools.install_gpu_profile` module from the exact rc4 source and therefore does not collect. The new rc5 terminal guide delegation is covered by `tests/test_r92_round_continuation.py`; this report does not claim a clean full legacy-suite run.

## Remaining human and scientific gates

The user must review the first card's selected candidates and supply real decisions before a human-owned model exists. At least one eligible CJ and one eligible non-CJ decision are required for the named small-data operational XGBoost fit. Optional biological YOLO training additionally needs independently completed CJ QA on eligible tiles from separate train and validation cards. The one-epoch synthetic detector has no biological-performance meaning. The two approved diagnostic photos are not an independent benchmark. No historical r92 refit, historical YOLO-on/off comparison, scientific accuracy, journal-submission readiness, or publication authorization is claimed.

The exact human resume and fit commands are in `FIRST_RUN_COMPLETE_ROUND_EN.md`. Private review state, fitted software-test models, detector weights, diagnostic images, and raw local logs stay outside the distributable source/wheel/sdist.
