# P2 — the recall guard fires only on a regression · result · 2026-09-24

Read against the terms fixed in
[`p2-recall-one-sided-2026-09-21.md`](p2-recall-one-sided-2026-09-21.md), filed as
[#132](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/132) before the
fix was committed and three days before any of this ran. Nothing trained, nothing
promoted.

**The change is not falsified: none of its three conditions fires.** Recall
improvements no longer raise an objection, PROMOTE at 5 se rises 7.6 pp, and no
REJECT and no verdict on a negative recall margin moves.

**The prediction at the MDE is falsified.** PROMOTE there rose to **40.5%**, not the
42–45% predicted. The change is not what went wrong. The derivation behind the
prediction was: see §3.

Reproduction:
`python pipeline/scripts/power_analysis.py --operating-curve --seed 7 --size-draws 6000`
and `--recheck`, each once against the pre-fix code and once against the fix.

## 1. What was run

- **Before** is `pipeline/src` at `974284e`, the fix's parent and byte-identical to
  `main` at `3f32fd2`. It was extracted with `git archive` and put first on
  `PYTHONPATH`. **After** is the one-line change in `8e1392e`. The imported
  `promotion.py` was checked both ways.
- `--operating-curve`: 6,000 draws, 5 v 5, the shipped band, only the gate-slice
  ROC-AUC displaced. Recall's true margin is zero in every cell. Each UNRESOLVED is
  attributed recall-first, as on 2026-09-20, and recall is now split by the sign of
  its margin. `blocked_contrast` is stubbed, as before.
- `--recheck`: every `promotion_log.json` on disk, re-decided against the champion
  summary it names with the tolerances it records. Nothing is stubbed and nothing is
  written.
- **The seed.** The pre-registration did not record one. The script's default, 42,
  does not reproduce its "now" column. **Seed 7 reproduces all 18 figures in #131's
  table exactly**, the nine verdict rates and the nine attribution shares, so it is
  the pre-registered baseline and the reading is paired against it. Seed 7 was
  found by probing seven seeds against the *baseline*, before any after-fix run.
  Seed 42 is in §4.

## 2. The result against the pre-registered terms

| true AUC margin | PROMOTE before | after | predicted | |
|---|---:|---:|---:|---|
| 0 — the null | 30.4% | **33.7%** | 33–35% | inside |
| 2.802 se — the MDE | 36.4% | **40.5%** | 42–45% | **missed, 1.5 pp low** |
| 5.0 se | 71.6% | **79.2%** | 78–80% | inside |

Seed 7, all six margins:

| true margin | before P / U / R | UNRES raised by: recall improved / regressed / auc | after P / U / R | recall improved / regressed / auc |
|---|---|---|---|---|
| 0 | 30.4 / 14.0 / 55.5 | 28.1 / 25.5 / 46.3 | 33.7 / 10.8 / 55.5 | 0.0 / 33.3 / 66.7 |
| 1.0 se | 36.2 / 38.4 / 25.4 | 16.3 / 16.9 / 66.8 | 39.7 / 34.9 / 25.4 | 0.0 / 18.6 / 81.4 |
| 1.33 se | 33.1 / 47.4 / 19.5 | 14.5 / 14.7 / 70.8 | 36.4 / 44.1 / 19.5 | 0.0 / 15.8 / 84.2 |
| 2.0 se | 28.4 / 58.3 / 13.3 | 13.0 / 12.9 / 74.1 | 31.6 / 55.0 / 13.3 | 0.0 / 13.6 / 86.4 |
| 2.802 se | 36.4 / 52.2 / 11.4 | 14.9 / 14.5 / 70.6 | 40.5 / 48.1 / 11.4 | 0.0 / 15.7 / 84.3 |
| 5.0 se | 71.6 / 17.2 / 11.3 | 45.6 / 44.0 / 10.4 | 79.2 / 9.6 / 11.3 | 0.0 / 78.9 / 21.1 |

**Condition 1, REJECT to PROMOTE: does not fire.** REJECT is identical in every cell
and the draws are paired by seed. That follows from the code as well: every REJECT
return in `evaluate_promotion` comes before the changed line, and none of them reads
its result. The one recorded REJECT is unchanged (§5).

**Condition 2, 5 se rises at least 5 pp: does not fire.** It rose 7.6 pp, from 71.6%
to 79.2%.

**Condition 3, a verdict on a negative recall margin changes: does not fire.** The
recall-regressed UNRESOLVED count, as a share of *all* draws, is unchanged in every
cell: 3.6, 7.6 and 7.6 pp at the null, the MDE and 5 se, before and after. Both
recorded UNRESOLVED verdicts have negative margins and are unchanged (§5).

The "recall improved" column goes to **0.0% in every cell**, which is the change
doing what it says.

## 3. The missed prediction, and why

The prediction treated the recall-improved share as what the fix would release. At
the MDE that share was 14.9% of 52.2% UNRESOLVED, **7.8 pp of all draws**. Only
**4.1 pp** became PROMOTE. The other 3.7 pp, **48%**, also carried an AUC objection
and stay UNRESOLVED. The recall-first attribution counted them once, as recall, and
so hid that.

This can be derived from the two tracked outputs. Only a draw that lost its recall
objection can move into the AUC column, and the AUC column rises from
70.6% × 52.2% = 36.9 pp of draws to 84.3% × 48.1% = 40.5 pp. The co-objection rate
depends on where the true margin sits against the band `[1.33 se, 3 se]`:

| true margin | recall-improved objections that also had an AUC objection |
|---|---:|
| 0 — the null | 18% |
| 2.802 se — the MDE, inside the band | **48%** |
| 5.0 se — above the band | 3% |

The MDE sits inside the band, so that is where the prediction failed. This is an
error in the derivation, not in the change. **Any prediction built from recall-first
attribution has the same bias**, and #128's pre-registration will be built from
exactly these tables.

## 4. Sensitivity: seed 42

| true AUC margin | before | after | rise | rise at seed 7 |
|---|---:|---:|---:|---:|
| 0 | 29.0% | 32.5% | +3.5 | +3.3 |
| 2.802 se | 35.1% | 39.2% | +4.1 | +4.1 |
| 5.0 se | 69.9% | 77.6% | +7.7 | +7.6 |

The rises agree to 0.2 pp. The levels move about 1.5 pp with the seed; the Monte
Carlo se near 30% is 0.6 pp. At seed 42 the null lands 0.5 pp under the predicted
range and 5 se 0.4 pp under. The predicted ranges were absolute levels built on one
draw of the baseline, and they were narrower than that baseline's own
seed-to-seed spread. **A pre-registration of a simulated rate should state the seed
and predict the paired change, not only the level.** The seed-42 reading of the MDE
agrees with seed 7's: missed.

## 5. Every recorded verdict, re-checked

| run | recorded | recall margin | direction | before the fix | after |
|---|---|---:|---|---|---|
| `phase1/arm-c-control` | UNRESOLVED | −0.0862 | regressed | UNRESOLVED | UNRESOLVED |
| `phase1/arm-d-difference` | UNRESOLVED | −0.1054 | regressed | UNRESOLVED | UNRESOLVED |
| `stage9/arm-d-difference` | REJECT | — | not reached | REJECT | REJECT |

None changes. **No recorded verdict has a positive recall margin**, so this re-check
can only confirm the side that was meant to stay the same. The bug never decided a
recorded verdict. Its effect is visible only in the simulation and in the unit test
that pins it.

## 6. The gate as it now stands

- **Size under a true null: 33.7%**, up from 30.4%, as §5 of the pre-registration
  predicted and accepted.
- **Power at the published MDE: 40.5%**, against the 80% that MDE asserts.
- **Still not monotone in the AUC margin**: 39.7% at 1 se, 31.6% at 2 se, 40.5% at
  2.8 se. That is #128's hole, which this change did not touch.
- **The recall guard is no longer the binding constraint.** At the MDE, 84% of
  UNRESOLVED verdicts are now raised by AUC. The recall objections left are
  regressions, the only kind it should raise.

## 7. Observed, not pre-registered: for #128

The same run's `(0, floor]` rows, the band P2.2e falsified, moved a long way:

| true margin | before | after |
|---|---:|---:|
| 0 | 0.1% | 1.0% |
| 2.802 se | 1.2% | 33.8% |
| 5.0 se | 1.8% | 42.9% |

Before the fix, about half of the UNRESOLVED verdicts behind P2.2e's 1.6% were
recall *improvements*. What is left is recall *regressions*, which are 83–100% of
that band's UNRESOLVED verdicts from the MDE up. Applied to recall, `(0, floor]`
calls any dip of up to one floor unresolved, and at 5 v 5 about 48% of candidates
have one under a true null. So #128's remedy still needs different rules for the AUC
side and the recall side. This is a pointer for that pre-registration, not a result.

## 8. What is not claimed

- Nothing about #128, #123 or `strict`.
- Not that the gate is well sized. It is 3.3 pp worse, as pre-registered.
- Nothing about a candidate whose recall *truly* improves. Recall's true margin is
  zero throughout, so this measures how the guard responds to noise in recall.
