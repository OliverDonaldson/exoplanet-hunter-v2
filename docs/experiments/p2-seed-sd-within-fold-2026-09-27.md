# P2 — the gate's seed sd, on the population it decides on · pre-registration · 2026-09-27

Written **before any change to `pipeline/src`**, per rule 6. Nothing here is adjusted
after the result. A result landing outside these terms is reported as falsified.
**No model is trained and nothing is promoted.**

Issue [#123](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/123).
This re-registers the change first pre-registered in
[`p2-seed-sd-population-2026-09-20.md`](p2-seed-sd-population-2026-09-20.md). That
file was falsified on its condition 3, size, because #128's hole made a wider floor
promote more. #128 landed on 2026-09-24
([result](p2-auc-floor-rule-result-2026-09-24.md)), so the size argument can now be
tested.

**One choice differs from 2026-09-20, and Ollie made it on 2026-09-27: the
estimator.** That file adopted the fold-pooled TESS sd, on M−1 df. This one adopts
the **within-fold TESS sd, variance-pooled over folds, on F(M−1) df**. §2b gives the
evidence the choice was made on.

## How this will be read, fixed before the work starts

It **succeeds** if the gate's AUC floor is built from a variance measured on the
population the margin is read on, if the implementation computes exactly the term
measured here, and if every change it makes to a verdict on disk is one listed in
§2c.

It is **falsified** by any one of:

1. **The implemented term differs from the study's.** For every run on disk with
   member scores, the new function must return §2a's `c4-corr` value to within
   1e-12, relative. The gap allowed is only the one between `math.lgamma` and
   scipy's `gammaln`.
2. **The gate decides differently from the emulation.**
   `power_analysis.py --recheck --corrected` against both champions must print the
   same output after the change as before it. So must `--options`, `--size`,
   `--operating-curve` and `--estimators`, at the seeds and draw counts in §5.
3. **Any verdict on disk changes in a way §2c does not list.** This includes a
   refusal §2c does not predict, and any recorded verdict changing at all.
4. **A multi-member summary without the new key is decided rather than refused**,
   or a single-member one is refused. The first is the silent fallback to the
   all-mission `seed_sd`, which is the defect. The second belongs to #137.

## 1. The defect

`decision_floor` builds the AUC floor from `variance.seed_sd`. That is the mean over
folds of the within-fold sd of per-member ROC-AUC on **all missions**
(`train.py:551`, aggregated at `:650-660`). The margin it bounds is the
**TESS-slice** ROC-AUC (`promotion.py:702-715`), because TESS is the deployment
population (`eval/scoring.py:296`). The champion's prior, `POOLED_SEED_SD = 0.0062`,
was checked on 2026-09-19 and is a TESS figure. So the mismatch is entirely on the
candidate side, and #123 §1 measured it.

This is the error-term rule from STAT 293 Part 1 §5. An F ratio is sized only when
the denominator's `E(MS)` matches the numerator's under H0. A floor built from
another population's variance has no reason to give the gate its nominal size.

## 2. What is already known, and is therefore not prospective

Everything here was measured before this file was written, from artefacts on disk,
with the tracked modes named. **None of it is a prediction.**

**2a. The term on every multi-member run.**
`power_analysis.py --recheck --corrected --against
models/cv/champion-rebaselined-today/cv_summary.json --against
models/reference/champion-m5-k2-2026-09-19/cv_summary.json`, on `main` at
`2322f6a`. Selected rows:

| run | M | `seed_sd` as recorded | fold-pooled TESS | within-fold TESS | df | **term, c4-corrected** | × recorded |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reference/champion-m5-k2-2026-09-19` | 5 | 0.00225 | 0.00372 | 0.00401 | 20 | **0.00406** | 1.80 |
| `cv/fc4f3515ee63…` (dual-view) | 3 | 0.00343 | 0.00733 | 0.00703 | 10 | 0.00721 | 2.10 |
| `cv/stage105-control` (branch) | 3 | 0.01411 | 0.00657 | 0.00850 | 10 | 0.00872 | **0.62** |
| `cv/stage8-synthetic` (branch) | 3 | 0.00818 | 0.01006 | 0.01297 | 10 | 0.01330 | 1.63 |

The full table has 23 rows. 21 are measurable, from 20 distinct predictions files,
because `fc4f3515-resummarised` reads its source's.

- **The re-baseline's term is 0.00406**, 1.80× what it records. That is #123's
  headline, reproduced on the adopted estimator.
- **The correction does not only widen floors.** Across runs, the corrected term is
  0.62× to 2.14× the recorded `seed_sd`. The 2026-09-20 file's condition 2 assumed
  it could only widen them. That holds for the dual-view runs, and not for every
  branch run.
- **Two runs cannot be measured.** `branches-20260807-shared` and
  `branches-20260808-capacity` record three members, but saved no member scores.

**2b. Why within-fold: the two estimators measure the same scale, and one has five
times the df.** From the same output, variance-pooled over runs of each
architecture:

| architecture | runs | fold-pooled | df | within-fold | df | ratio | F test |
|---|---:|---:|---:|---:|---:|---:|---|
| `cnn_dualview` | 3 | 0.00513 | 8 | 0.00526 | 40 | **0.98** | F = 0.95, p = 0.97 |
| `cnn_branches` | 17 | 0.01108 | 34 | 0.01294 | 170 | **0.86** | F = 0.73, p = 0.29 |

What #123 got wrong was the **population**, all missions, not the protocol. On
TESS, the within-fold and fold-pooled estimators do not differ detectably in scale.
The fold-pooled one has M−1 df: 2 at the three members the weekly refresh trains,
and 4 at five. Here is what that df costs the gate.
`power_analysis.py --estimators --seed 7 --size-draws 6000`. Each simulated arm's
recorded term is an unbiased estimate drawn on its estimator's df.

| allocation | estimator | size | power at the MDE |
|---|---|---:|---:|
| 5 v 5 | exact term | 1.8% | 63.7% |
| | fold-pooled, M−1 df | 2.5% | 60.0% |
| | **within-fold, F(M−1) df** | **2.1%** | **63.2%** |
| 3 v 3, the refresh | exact term | 1.7% | 57.4% |
| | fold-pooled, M−1 df | 3.1% | 50.4% |
| | **within-fold, F(M−1) df** | **1.8%** | **55.1%** |

Seed 42 orders the estimators the same way in all three allocations. Rows are not
paired, because they draw different numbers of variates. **The within-fold estimator
is closer to exact on both size and power in every allocation.** The price is that
for branch models it may overstate the floor, by up to 16% on the point estimate.
That errs conservative, and the difference is not significant.

**2c. Every verdict on disk, with the term read.** The same command, compared with
`--recheck` without `--corrected`:

| champion | run | before | after | why |
|---|---|---|---|---|
| rebaselined-today | `stage8-synthetic` | PROMOTE | **UNRESOLVED** | floor 0.01559 → 0.01974; margin 1.009× → 0.797× |
| rebaselined-today | `fc4f3515-resummarised` | PROMOTE | **UNRESOLVED** | it had no AUC floor; it gains 0.01494, and the margin is 0.460× |
| rebaselined-today | `branches-20260807-shared` | REJECT | **refused** | three members, no member scores |
| both | `branches-20260808-capacity` | REJECT | **refused** | three members, no member scores |
| M=5 re-baseline | `branches-20260807-shared` | REJECT | **refused** | three members, no member scores |

**No other verdict changes, and no recorded verdict changes.** Two notes on that
list:

- **The closest call is one that does not move.** `stage105-control`'s floor
  *narrows*, from 0.02048 to 0.01597, which takes its margin from 0.73× to 0.94× the
  floor. It stays UNRESOLVED.
- `stage8-synthetic` is the promotion #128's result disclosed as the cost of landing
  before this change. This change takes it back.

**2d. Where this reaches #137.** `fc4f3515-resummarised` gains a floor because the
new key is written where `summarise_scored` builds its variance block. That block
never carried an AUC `seed_sd`. So re-summarised multi-member runs stop falling
through #137's missing-floor path. **Single-member candidates are unchanged**, and
#137 stays open for them.

## 3. The estimator

For the gate slice, TESS:

1. **Per fold and per member**, compute ROC-AUC on that fold's held-out TESS rows.
2. **Per fold**, take the variance across members, with ddof = 1.
3. **Pool across folds** by averaging the variances, with df = F(M−1). Members per
   fold are equal, so the weights are.
4. Take the square root, and **c4-correct on the pooled df**: divide by
   `c4(df + 1)`, which is 0.9754 at 10 df and 0.9876 at 20.

The steps follow three rules:

- **Pool variances, not sds.** `train.py:658` averages sds, which understates σ. So
  does leaving the result uncorrected.
- **c4-correct, as the 2026-09-20 file adopted.** At 10–20 df the correction is
  1–3%.
- **Recall is not touched.** `pooled_gate_recall_seed_sd` is fold-pooled on M−1 df
  and uncorrected, so it carries the same thin-df cost at M=3. It is recorded here
  (§6) and not changed. Changing it would move the recall floor, which is a
  different decision.

## 4. What will be changed

1. **`eval/comparison.py`**: a function beside `pooled_member_draws` computes §3
   from a predictions frame. It needs `fold`, `label`, `mission` and the member
   columns. It returns `gate_roc_auc_seed_sd` and `gate_roc_auc_seed_df`, and
   `None` with 0 df below two members. **It raises** on a non-finite member score,
   a missing `fold` column with two or more members, or a single-class fold slice,
   as `pooled_member_draws` already does for recall. c4 is computed with
   `math.lgamma`, because a new scipy import in `pipeline/src` breaks the mypy
   baseline.
2. **The three sites that already merge `pooled_member_draws`** into the variance
   block also merge this: `train.py:668`, `train_branches.py:672` and
   `eval/scoring.py:309`. So fresh runs from either trainer carry it, and so does
   anything re-summarised.
3. **`promotion.py`**: `decision_floor` reads `gate_roc_auc_seed_sd` for the AUC
   term, on both sides.
   - A summary recording **two or more members** without that key, or with it
     `None`, **raises**, naming the run and how to re-summarise it (rule 8).
     Falling back to `seed_sd` would be the defect itself.
   - One member, or no variance block, is unchanged. That is #137's path.
   - A champion with no variance still borrows `POOLED_SEED_SD`, unchanged.
4. **`variance.seed_sd` keeps its value and its writers.** The record reads it.
   Its docstring gains the population it describes.
5. **Tests.** Fixtures that carry `seed_sd` for two or more members also carry the
   new key at the same value, so no fixture's floor moves. New tests pin:
   - the estimator against a hand-computed case;
   - the refusal of a stale multi-member summary;
   - that `seed_sd` alone is no longer read.

**What it costs operationally.** Every multi-member `cv_summary.json` already on
disk lacks the key, so after the change the gate refuses them until they are
re-summarised. That includes the M=5 re-baseline and both Phase 1 arms. Nothing on
disk is rewritten by this change. Re-summarising the re-baseline, to a new path,
is the step that makes it usable as champion.

The weekly refresh is not affected:
- its candidate is trained fresh, so it carries the key;
- its champion comes from the control lane, which is single-member, so it borrows
  the prior as before.

## 5. How the result will be read — the prospective part

**5a. The term.** For each of the 21 measurable rows in §2a, call the new function on
the frame `tess_seed_sd` reads, and compare with its `corrected` value. Condition 1.

**5b. The gate.** Run each of these before and after the change, and diff the
outputs:

- `--recheck --corrected` against both champions;
- `--options --seed 7 --size-draws 6000`;
- `--size --seed 7 --size-draws 6000`;
- `--operating-curve --seed 7 --size-draws 6000`;
- `--estimators --seed 7 --size-draws 6000`.

The simulated arms write `seed_sd` and the new key at the same value, so every one
of them must be byte-identical. Condition 2.

**5c. The refusal, on real data.** After the change, `--recheck` *without*
`--corrected` must raise on the first recorded log, `phase1/arm-c-control`, whose
summary records three members and no new key. Condition 4. `promotion_gate.py` is
not run on a real run directory, because a wrong implementation would overwrite the
log it writes there.

**5d.** `make test`, with the tests in §4.5.

## 6. What is not claimed

- **That the fold-pooled and within-fold scales are equal.** They are not
  detectably different, on 20 runs of two architectures.
- **That recall's term is right.** It is fold-pooled on M−1 df and not
  c4-corrected. At M=3, `E[s] = 0.886σ`, so it understates by about 11% in
  expectation, before its df noise. Left for its own decision.
- **Anything about single-member candidates** (#137), the Brier and ECE constants
  (#134), or the console's displayed floor. `api/app/routes/model.py:95` still
  reads `seed_sd`. That is a display, and it is inert while the served champion is
  single-member.
- **That the gate is then correctly sized end to end.** At 5 v 5 its size is 2.1%
  on this estimator, and its power at the MDE is 63.2%, against the 81% ceiling
  from #134.
