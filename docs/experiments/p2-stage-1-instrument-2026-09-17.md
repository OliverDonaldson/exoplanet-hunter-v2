# P2 Stage 1 — the instrument, before another model is trained · 2026-09-17

`PLAN.md` §5, superseding **P2.2 as written**. Four defects in how this project
decides, found by crossing the VUW course notes against the repo and measured
against artefacts already on disk. **Nothing was trained, nothing promoted,
no DVC command run.** Issues
[#93](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/93),
[#94](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/94),
[#95](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/95),
[#98](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/98).

Reproduction: `python pipeline/scripts/power_analysis.py --n-boot 2000`.

## How this will be read, fixed before the work finished

Pre-registered, per rule 6. Stage 1 changes no model and can therefore improve
no metric. It succeeds if **the gate's floor is derived from a measurement
rather than an assumption, and the derivation is checkable**. Three things had
to come out of it, and any one of them failing is the falsifying outcome:

1. A seed sd on more than 2 df, **split by architecture**, from runs already
   written. If the pooled estimate had landed within the 2-df estimate's noise,
   the extra df would have bought nothing and the item would be recorded as
   unnecessary.
2. The corrected floor **re-checked against every recorded verdict**. A verdict
   that changes is reported as a change, not rewritten (rule 5).
3. A blocked test that **runs**. If the fold x member layout had not been on
   disk, the item would be deferred to P2.6 rather than approximated.

All three landed. The corrections are appended under
[P2.1](p2-1-power-analysis-2026-09-14.md), which is not edited.

## 1. The seed sd was one arm's, on 2 df, from the wrong architecture

`seed_spread` read arm C's three members only. Fifteen multi-member runs were
already on disk. Per-run ROC-AUC seed sd across them spans **0.0032 to 0.0193**,
a 6x range — what 2 df looks like. Pooled on the TESS gating slice:

| architecture | pooled seed sd, ROC-AUC | runs | df |
|---|---:|---:|---:|
| `cnn_branches` | **0.0108** | 13 | 26 |
| `cnn_dualview` | **0.0062** | 2 | 4 |
| all pooled | 0.0103 | 15 | 30 |

Arms C'' and D'' are `cnn_branches`. Their sd was the project's MDE for every
comparison including P2.6's, which compares both — the category error
`docs/index.md` rule 7 forbids. Architecture is now read from each run's member
checkpoint names and **raises** when it cannot be determined.

Decision [#34](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/34)
judged that "roughly ten draws would" retire the thin-floor limitation and that
the research was frozen. Thirteen extra draws needed no research.

## 2. The MDE carried one seed term where a contrast needs two

P2.1 computed `hypot(boot, sd_seed / sqrt(M))`, the sd of one arm's mean.
`decision_floor` has read `sqrt(sd_cand^2/n_cand + sd_inc^2/n_inc)` since 4.1b.
The gate and the power analysis disagreed about the same quantity. Corrected,
at 80% power, TESS ROC-AUC:

| comparison | M=3 | M=5 | M=10 |
|---|---:|---:|---:|
| branch vs branch | 0.0255 | **0.0201** | 0.0148 |
| dual-view vs dual-view | 0.0155 | **0.0126** | 0.0098 |
| *as P2.1 published* | 0.0162 | *0.0131* | 0.0101 |

**0.0131 is about right for a dual-view challenger and too lenient by 1.5x for
a branch one.** Both columns are printed, so the published figure stays
locatable beside the correct one.

**Pairing on member index does not help.** If the arms' seed draws were shared,
no second term would be needed. Measured: the member-paired contrast sd is
**0.0175** against `sqrt(2) x 0.0093 = 0.0130` predicted under independence, and
the member main effect common to both arms is ~0. Recorded because it rules out
the obvious fix before compute is spent on it.

## 3. The gate divided the champion's borrowed variance by the candidate's members

`decision_floor` fell back to `pooled**2 / n_models` — the *candidate's* count —
where 4.1b specifies `n_inc`. The served champion is a single-member run, so
every gate decision this project has made read a floor understated by ~**1.9x**
at five members. `n_inc` is now `CHAMPION_MEMBERS_WHEN_UNMEASURED = 1`, named in
`floor_source`, and pinned by a test.

`POOLED_SEED_SD` moves **0.0081 -> 0.0062**. The provenance is worth tracing,
because it shows the error was structural rather than a slip:
[stage 4](stage-04-branch-runs.md) measured `seed_sd 0.0081` on the **branch**
model over three members; [stage 6](stage-06-recall-floor.md) adopted
`2 x 0.0081 / sqrt(3) = 0.0094` as the AUC threshold from that; and the constant
then became the *champion's* prior — for a champion that has always been
dual-view. A branch number has stood in for a dual-view one since the constant
was written, which is exactly what 4.1b line 338 refused when it rejected
pooling across architectures.

**Every recorded verdict re-checked. None changes.**

| run | verdict | recall floor as recorded | corrected | margin vs corrected |
|---|---|---:|---:|---|
| `phase1/arm-c-control` | UNRESOLVED | 0.0792 | 0.0979 | -0.0862, 0.9x — inside the 1.5x band |
| `phase1/arm-d-difference` | UNRESOLVED | 0.0718 | 0.0921 | -0.1054, 1.1x — inside the band |
| `stage9/arm-d-difference` | REJECT | 0.0623 | 0.0849 | population mismatch, upstream of the floor |

The correction widens floors, so it can only move a REJECT toward UNRESOLVED,
never toward PROMOTE. It did not move either.

## 4. The one blocked test in the gate could never run

`MIN_PAIRS_FOR_P_VALUE = 6` against `n_splits: 5` in all three model configs.
`paired_folds`' docstring states the blocking argument correctly; its p-value
branch was unreachable from 2026-08 until now.

**This supersedes a recorded adoption, and says so rather than quietly dropping
it.** [audit-2026-08-07](audit-2026-08-07.md) adopted the paired Wilcoxon
deliberately, with the right reasoning — at five folds it floors at p=0.0625, so
a gate keyed on it would reject every real improvement — and
[standing-audits](standing-audits.md) counts it among the six adoptions done.
What is retired is the **p-value path**, not the pairing: `paired_folds` still
reports the paired deltas, the win count and Cohen's *d*, and the blocking
argument its docstring makes is the argument `blocked_contrast` acts on. The
2026-08-07 reasoning was correct about the test and incomplete about the remedy
— five folds is the wrong replication unit, not an insurmountable n.

`blocked_contrast` reads the (fold x member) layout `cv_summary.json` already
writes as `model_roc_auc`. **Fold is a block** — both runs held out the same
rows — and **member is nested in arm**, so the F denominator is the
member-within-arm mean square on `a(M-1)` df, not the residual. Testing against
the residual is the nested-design trap: its expectation is missing the member
term the numerator carries, so it rejects far too often. The p-value is a
**within-block permutation test**: arm labels are exchangeable within a fold
under the null and nowhere else.

On the Phase 1 arms, 5 folds x 3 members:

| metric | contrast | 95% CI | F(1,4) | p |
|---|---:|---|---:|---:|
| ROC-AUC | -0.0032 | [-0.0238, +0.0175] | 0.18 | 0.72 |
| gate recall @1% FPR | -0.0396 | [-0.1594, +0.0802] | 0.84 | 0.42 |

**This is stricter than the reading quoted in #95** (t(8) = -2.17, p = 0.062),
which treated member index as pairing across arms. §2 measured that it does not,
so the nested denominator is the defensible one and the earlier figure is
withdrawn. The correction runs against the finding's own author; it is recorded
rather than quietly dropped.

**It returns None, not an approximation, when either run has one member per
fold** — which is every comparison against the current champion. P2.6 at five
members is what makes this fire against a served model. Until then the gate's
strongest reading is available only arm-to-arm.

## 5. The gating metric's margin is carried by a handful of rows

Jackknife leave-one-out influence on the contrast, TESS slice:

| metric | contrast | top 1 row | top 5 | top 10 | rows to flip the sign |
|---|---:|---:|---:|---:|---:|
| ROC-AUC | -0.0016 | 0.26x | 1.09x | 1.88x | **6** |
| pAUC FPR<=0.1 | -0.0056 | 0.34x | 1.49x | 2.53x | **4** |
| recall @1% FPR | -0.0146 | **1.21x** | 4.79x | 8.74x | **1** |

**One row reverses the recall contrast.** P2.1 §3's demotion, from a second
direction. The count scales with the contrast, and this one is ~0.13x the MDE,
so a small number here is not on its own a strong claim — the script says so in
its own output.

## 6. What Stage 1 does not fix

- **The dual-view sd is still only 4 df.** Better than 2, not good. P2.6 at five
  members over both architectures is what retires it.
- **`blocked_contrast` cannot read the TESS slice for AUC.** The per-member
  fold rows carry a gate-sliced key for recall only, so the AUC member axis is
  pooled over missions. Stated rather than worked around.
- **No metric gained power in the follow-up region.** P2.1 §4 stands untouched;
  only [#97](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/97)
  addresses it.
- **The bar moved against the project, not for it.** Every correction here
  widens a floor or strengthens a denominator. Nothing became easier to promote.
