# The record

What was measured, in the order it happened, one file per stage or audit. Every
file was moved verbatim from `docs/roadmap.md` on 2026-09-04 and is frozen: a
correction is a dated note appended under the entry it corrects, never an edit.
Pre-registrations sit immediately before the result they fixed the reading of.
The old-to-new stage numbering is at [`../roadmap.md`](../roadmap.md) §1c, and
every old section number is a pointer there.

Two conventions run through all of it. A margin is read against the noise floor
measured in the same run, by the rule 2·sd/√n over members per fold, and a floor
belongs to the architecture it was measured on. A result landing outside the
terms fixed before it ran is recorded as falsified, never re-specified. The
outcome column below is each entry's own reading of itself, never a later
reinterpretation. Each row was drafted from the file, then checked against it by
two independent adversarial readers, one on the numbers and one on the claims;
the fourteen rows they flagged were corrected by hand against the file before
this index was published.

| file | covers | dates | outcome | one line |
|---|---|---|---|---|
| [`stage-01-03-inputs-and-rebaseline.md`](stage-01-03-inputs-and-rebaseline.md) | stages 1–3 · §3.1 | 2026-08-05 to 2026-08-08 | mixed | Stages 1–3 done: an eleven-branch 5,423-example view set and a 3.6 GB DV archive, shards 2.6x larger at 301/31; the momentum-dump branch and 33x33 stamps could not be built as specified, and the re-baselined per-mission summary is what lets the gate decide. |
| [`audit-2026-08-07.md`](audit-2026-08-07.md) | audit of the recorded numbers · §2b–2e | 2026-08-07 to 2026-08-08 | audit | Every run-1 number reproduces, with one misprint corrected and one false claim about the TESS low-count cells retracted; run 2 lost all five paired folds (mean −0.0313, pairing flagged inexact) and every re-baselined mission; the TESS gate had never engaged. |
| [`stage-04-branch-runs.md`](stage-04-branch-runs.md) | stage 4 · §3.2 | 2026-08-05 to 2026-08-08 | rejected | Every branch arm rejected on TESS recall @1% FPR (0.238, 0.126, 0.145, 0.236 vs 0.307); resolution hypothesis falsified (Kepler +0.0707) and capacity closed (paired -0.0035, d=-0.44). |
| [`stage-05-viewset.md`](stage-05-viewset.md) | stage 5 · §3.3 | 2026-08-08 | confirmed | Candidate view set rebuilt cold as budgeted — 5,346 rows at 2001/201, 309 MB, 95 minutes — well-formed and scoreable by run 3's fold-0 checkpoint; the entry also re-specifies stage 7's criterion and plans stage 6. |
| [`stage-06-recall-floor.md`](stage-06-recall-floor.md) | stage 6 · §3.4 | 2026-08-08 to 2026-08-09 | confirmed | The recall noise floor is measured: pooled seed sd 0.0292 gives a ~0.034 decision floor, run 3's rejection is sound at 4.8x it, and the capacity arm's gain was real but stays unactionable at 0.220 against the champion's 0.307. |
| [`execution-order-2026-08-09.md`](execution-order-2026-08-09.md) | execution order and impact ranking · §2f–2g | 2026-08-08 to 2026-08-09 | decision | Execution order fixed as 7i→8→9→10→7ii→11→12 by splitting stage 7 into instrument and reading, removing all three backward edges; integers kept as names; stage 8 ranked highest impact, stage 7 lowest. |
| [`readiness-contract-2026-08-09.md`](readiness-contract-2026-08-09.md) | the readiness contract · §2h | 2026-08-09 | decision | Readiness after stage 11 judged yes with two exceptions: an unexplained +0.1446 Kepler cell and no published catalogue; item 1 likely resolves by closing the branch line, not promotion. |
| [`stage-07-attribution-sweep.md`](stage-07-attribution-sweep.md) | stage 7, the attribution sweep · §3.5 | 2026-08-09 | mixed | Stage 7's 26.4% host-pass criterion is unmeasurable as written and was re-specified; the three-family leave-one-out sweep read null on every arm, and spending no further stage-7 compute before stage 8 is recorded as a recommendation, not a decision taken. |
| [`stage-07i-control-arm-harness.md`](stage-07i-control-arm-harness.md) | stage 7i · §3.6 | 2026-08-09 to 2026-08-12 | null | Stage 7's criterion is not met: on 580 baseline-matched hosts the branch model has no measured advantage over the incumbent on either criterion, and the one prediction the file calls falsified is falsified at its own F1 cut and not at the other operating point. |
| [`gate-calibration-2026-08-12.md`](gate-calibration-2026-08-12.md) | promotion gate audit · §3.7 | 2026-08-12 | audit | Three gate defects found and fixed, none of which changes a recorded verdict — only what a verdict was allowed to claim; `paired_folds` stays computed but never gating and is recorded as a known limitation rather than fixed. |
| [`observation-baseline.md`](observation-baseline.md) | observation baseline · §3.8 | 2026-08-05 to 2026-08-12 | audit | Score–baseline correlation is a genuine label confound (label +0.387 on TESS); the branch model amplifies it to +0.5155, +0.13 above the labels, so the fix is label-distribution intervention, not architecture. |
| [`stage-08-labels-and-negatives.md`](stage-08-labels-and-negatives.md) | stage 8 · §3.9 | 2026-08-08 to 2026-08-15 | mixed | Propensity weighting removed the branch model's amplification of baseline dependence (gap −0.1336, 3.3× bar) at no recall cost; synthetic null, arm S unreadable, labels untouched, all four predictions falsified, split win qualified. |
| [`ui-scaffold-audit-2026-08-13.md`](ui-scaffold-audit-2026-08-13.md) | UI scaffold audit · §3.10 | 2026-08-13 | audit | Scaffold audit against the API: a fourth tab was added for the branch evidence stage 11 will produce, the probability-vs-epoch panel was identified as unphysical with `per_fold` and `prob_std` named as the better versions already on the contract, and two computed quantities the scaffold never showed were listed. |
| [`stage-10-5-ensemble.md`](stage-10-5-ensemble.md) | stage 10.5 · §3.11 | 2026-08-12 to 2026-08-15 | mixed | Both ensemble arms clear their bars on shortlist recall (0.4362 and 0.4223 against the common-fold dual-view member's 0.3046), so the branch line's value is as a complement; on host-scoring, the defect W2 names, the branch models are better than the incumbent by a margin excluding zero, and stage 8's qualified second win is recorded as not to be banked. |
| [`forward-plan-2026-08-16.md`](forward-plan-2026-08-16.md) | the forward plan as of 2026-08-16 · §4 intro, §4.3–4.8 | 2026-08-14 to 2026-08-20 | plan | The forward plan, not an experiment: the remaining items re-costed at ~45–60 h plus an unestimated UI, with the probable ending recorded as the branch line closed in writing and `ca906040`, or its stage 10 retune, staying served. |
| [`refresh-gate-calibration-4-1.md`](refresh-gate-calibration-4-1.md) | refresh gate calibration · §4.1–4.1d | 2026-08-15 to 2026-08-17 | mixed | Refresh gate calibrated: recall floor measured at 0.0733 (3.7x the 0.02 legacy constant); control lane wired after its first check was falsified and replaced; weekly loop had never been able to promote; nothing promoted. |
| [`stage-09-difference-image.md`](stage-09-difference-image.md) | stage 9 and the ExoMiner++ readout · §4.2, §4.2b | 2026-08-17 to 2026-08-20 | unmeasurable | Difference-image branch built and run; its W2 falsification test (control-arm host-AUC) is unmeasurable because the stage 7i harness zeroes the branch on control rows; predictions 2–4 confirm it costs nothing, none show it delivers anything. |
| [`phases-pre-registration-4-2c.md`](phases-pre-registration-4-2c.md) | the four phases, pre-registered · §4.2c | 2026-08-20 | plan | Pre-registration only, nothing run: fixes phase order (console first, seed-sweep floors before Phase 1 is read or stage 10 starts) and the falsification thresholds each phase must meet; no promotion. |
| [`phase-1-build-4-2d.md`](phase-1-build-4-2d.md) | Phase 1 and Phase 1a build · §4.2d | 2026-08-27 to 2026-08-28 | built-not-run | Target-position channel and momentum-dump view built for Phase 1; neither arm has run. Box centre misses the target's pixel on 77.8% of stamps; stage-9 dv_usable baseline recorded (D−C −0.0075, 0.33x floor). |
| [`standing-audits.md`](standing-audits.md) | standing audits · §5 | 2026-08-07 to 2026-08-15 | audit | Four audits done and acted on: ExoMiner delta 6/10, security clean with rate limiting added and protobuf CVE accepted, nothing to delete, DVC pointer move affects no measured number. |
| [`change-log.md`](change-log.md) | change log · §7 | 2026-08-14 to 2026-08-17 | audit | Record restructure verified mechanically lossless; Champion rename left pre-registrations, recorded results and the DVC path unchanged; docs consolidated to eight, W14 closed; 3.11c's reproduction script untracked, pre-run authorship unverifiable. |

## Adding to the record

A new stage gets a new dated file here and one row above. A correction to an
existing file is a dated note under the entry it corrects. Nothing here is
rewritten, renumbered or summarised in place.
