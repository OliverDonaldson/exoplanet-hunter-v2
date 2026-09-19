# P2 — the gate's seed sd, on the population it decides on · pre-registration · 2026-09-20

Written **before any code changes**, per rule 6. Nothing here is adjusted after
the result; a result landing outside these terms is reported as falsified.

This changes the gate's arithmetic. **No model is trained and nothing is
promoted.** `models/registry.json` is not touched and the served champion stays
`ca906040`.

Issue [#123](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/123).
Supersedes the floor and MDE figures in
[#114](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/114) and in
[`p2-champion-rebaseline-2026-09-19.md`](p2-champion-rebaseline-2026-09-19.md).

## How this will be read, fixed before the work starts

It **succeeds** if the gate's AUC floor is derived from a variance measured on
the same statistic and the same population as the margin it bounds; if the
choice among defensible estimators of that variance is made on stated evidence
rather than convenience; and if the gate's size is measured under a true null
both before and after the change.

It is **falsified** by any one of:

1. **The corrected term is not computable for the runs the gate must decide
   on** — if it needs artefacts the trainers or `summarise_scored` do not write,
   the fix is not deliverable as specified and is recorded so, not approximated.
2. **Correcting the candidate's population moves a recorded verdict toward
   PROMOTE.** Held at the current champion, this correction raises the candidate
   term (0.00225 → 0.00372 on the re-baseline) and therefore only widens floors.
   A widened floor can move PROMOTE toward UNRESOLVED and UNRESOLVED toward
   REJECT, never the reverse. A verdict moving the other way means the diagnosis
   is wrong.
3. **The gate's measured size moves further from nominal after the correction
   than before.** If matching the error term to the estimand does not improve
   the gate's size, the argument in §4 is wrong and the change is not justified
   by it.

## 1. The estimand

`evaluate_promotion` compares `per_mission[GATE_MISSION]["roc_auc"]`
(`promotion.py:702-715`) — the **fold-pooled out-of-fold TESS** ROC-AUC.
`eval/scoring.py:296` fixes why: *"TESS is 100% of the deployment population"*.

The floor that bounds a margin in that quantity must be built from the
member-to-member standard deviation **of that same quantity**. It currently is
not: `variance.seed_sd` is the mean over folds of the within-fold spread of
per-member **all-mission** ROC-AUC (`train.py:551`, aggregated at
`train.py:650-660`).

## 2. What is already measured, and is therefore not prospective

Stated plainly because rule 6 is about not re-specifying after the fact. Every
number in this section was measured **before** this file was written, from
artefacts already on disk, and is reported in #123. It motivates the change; it
does not test it.

Per-member ROC-AUC seed sd on `models/reference/champion-m5-k2-2026-09-19`
(M=5, 5 folds, 5,378 rows after the mission join):

| protocol | population | seed sd |
|---|---|---:|
| within-fold | all-mission | **0.00225** — what `variance.seed_sd` records |
| within-fold | TESS | 0.00390 |
| fold-pooled | all-mission | 0.00238 |
| fold-pooled | TESS | **0.00372** — the gate's own statistic |

`POOLED_SEED_SD = 0.0062` was checked rather than taken from its docstring, and
is confirmed a TESS figure: the two `cnn_dualview` multi-member runs it pools
give TESS within-fold sds of 0.00494 (`510b565b9f7f…`) and 0.00625
(`fc4f3515ee63…`), against all-mission 0.00302 and 0.00343. **The mismatch is
entirely on the candidate side.**

## 3. The choice this fixes, and the rule that fixes it

Three defensible estimators of "the TESS seed sd" differ by 1.09x, and picking
one silently is how the original defect happened.

| estimator | value | df at M=5 |
|---|---:|---:|
| mean of within-fold sds (the current convention, on TESS) | 0.00390 | ~20 |
| RMS of within-fold sds, c4-corrected | 0.00427 | 20 |
| fold-pooled TESS sd — matches the estimand exactly | 0.00372 | 4 |
| fold-pooled TESS sd, c4-corrected | 0.00396 | 4 |

**Adopted: the fold-pooled TESS sd, c4-corrected.** The rule, fixed here:

- **Match the estimand first.** The fold-pooled estimator measures the spread of
  the statistic the gate actually reads. The within-fold estimator measures the
  spread of a different statistic on a fifth of the rows. STAT 293's nested
  designs make the cost of the substitution explicit — the denominator must be
  the mean square whose `E(MS)` matches the numerator's under H0, or the ratio
  has no reason to be correctly sized under H0.
- **Correct the small-sample bias of `s`.** `E[s] = c4(M)σ`, and `c4(5) = 0.9400`,
  so an uncorrected sd understates σ by 6% at M=5. The current code also averages
  standard deviations rather than pooling variances, which compounds it.
- **Accept the df cost and record it.** 4 df is thin. The within-fold estimator
  buys ~5x the df, and across the three multi-member dual-view runs on disk the
  fold-pooled : within-fold ratio is 0.95, 0.99, 1.17 — no consistent sign, so
  no usable bias correction, only extra variance of unknown direction. A thin
  estimate of the right quantity is preferred to a tight estimate of the wrong
  one; **the thinness is recorded as a known limit, not resolved here.** M=10
  would take it to 9 df and is the standing route.

## 4. What will be changed

1. `eval/comparison.py` — `pooled_member_draws` already re-forms a per-member
   out-of-fold TESS set for recall. The AUC statistic is the same shape on the
   same rows: it emits `pooled_gate_roc_auc`, `pooled_gate_roc_auc_seed_sd`
   (c4-corrected) and `pooled_gate_roc_auc_n_draws` beside the recall keys.
2. `promotion.py` — `decision_floor` reads the new key for the AUC term on both
   sides. When a summary carries a variance block but not the new key, it
   **refuses** with a named reason rather than falling back to `seed_sd`
   (rule 8): a silent fall-back to the all-mission key is the defect itself.
3. `train.py` — `variance.seed_sd` is **left as it is**. It is the all-mission
   within-fold figure, the record reads it under that meaning, and rule 5 keeps
   the record's numbers legible. Its docstring gains the population it describes.
4. `POOLED_SEED_SD` — value unchanged; it is already TESS. Its docstring gains
   the pooling convention and the two runs it came from.

**Backfill, and why it matters.** Because the new key is computed in
`pooled_member_draws`, it lands on both paths — the trainers and
`summarise_scored` — so every multi-member run on disk gains it by re-scoring
alone. That is the property `seed_sd` does not have and which forced #114's
retrain. It does **not** give the served champion one: `ca906040` is
single-member, so it has no seed spread to measure, and the borrowed prior
stands until a multi-member champion exists.

## 5. How the result will be read — the prospective part

Two studies, in order. Neither trains a model.

**5a. Every recorded verdict, re-checked.** Each multi-member run in the record
is re-summarised to obtain the new key, and `evaluate_promotion` re-run against
the champion it was originally read against. Reported as a table of verdict
as-recorded against verdict-corrected. Falsification condition 2 applies.

**5b. The gate's size under a true null — [#115](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/115), now the acceptance test rather than a separate errand.**
`evaluate_promotion` is a deterministic function of two summary dicts, so member
metrics are drawn at the measured sd with µ_c = µ_i, synthetic summaries
assembled, and verdicts counted over 10,000 draws. Run under the **current**
term and the **corrected** term, non-strict and strict, reporting REJECT /
UNRESOLVED / PROMOTE rates separately. Falsification condition 3 applies.

**Prediction, recorded before 5b runs.** The floor is `2 x se` and the
UNRESOLVED band is `[floor/1.5, floor x 1.5]` = `[1.33 se, 3 se]`. Under a true
null `|δ̂|/se` is half-normal, so **82% of draws land below the band**, roughly
half of them positive — and `promotion.py:886-917` reaches PROMOTE for a
positive margin below the band once no other criterion has rejected. If the
other criteria do not catch them, the non-strict false-PROMOTE rate is of order
**40%, not of order 2%**, and correcting the seed sd will not fix that: it is
the band's geometry, not the floor's width. Recording it here so that the
measurement can contradict it.

## 6. What is not claimed

- No performance claim. Nothing here makes any model better.
- The 0.0088 MDE at 5v5 on the corrected term is **not** the bar the gate faces
  today. It requires a multi-member champion, which requires #114's re-baseline
  to be adopted, which is a separate decision and is not taken here.
- Wiring the re-baseline in as champion **unchanged** would give an AUC floor of
  0.00285 against the 0.00471 the corrected term supports — 1.65x too narrow.
  That is the specific trap this pre-registration exists to close, and it is not
  closed until item 2 of §4 lands.
