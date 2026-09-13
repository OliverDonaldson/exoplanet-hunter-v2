# The centroid flag, measured against labels · 2026-09-13

Commissioned by [#22](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/22), which says plainly:
*"a flag that fires on a quarter of the population is not a flag. Measure
against labels before changing it."* This is that measurement. **The threshold
was not changed**; the reason is at the end.

## What was measured

`centroid_snr` is the quadrature sum of the per-axis in-transit centroid shifts
in units of the out-of-transit scatter (`features/centroid.py`). `/score` serves
`suspicious = centroid_snr > BEB_THRESHOLD_SIGMA` with the threshold at **3.0**,
and the console renders that as the Centroid Shift card: *"In-transit centroid
stays on the target star"* against *"Centroid offset in transit; flux may come
from a neighbour"*.

Source: `data/processed/views.npz`, `aux_features[:, 8]` (the `CENTROID_COL`
index, raw and unimputed) against `labels` — **5,355 finite rows of 5,380**,
2,866 confirmed and 2,489 false positives. Bulk-scored candidates come from
`results/candidates_scored.parquet` (4,685 rows).

## Result 1 — the statistic carries real signal

| | n | median | p75 | p90 | > 3σ |
|---|---:|---:|---:|---:|---:|
| confirmed | 2,866 | 1.52 | 3.24 | 6.20 | **27.6%** |
| false positive | 2,489 | 5.05 | 34.34 | 277.44 | **61.8%** |

**AUC for predicting "false positive" = 0.7339.** The direction is the physical
one and the separation is well clear of chance. Whatever else is wrong here, the
quantity is not noise, and #22's "either the statistic or the threshold is
miscalibrated" is answered: the statistic is informative.

## Result 2 — the threshold is miscalibrated

At 3.0σ the flag fires on **43.5% of the labelled population** and on **27.6% of
confirmed planets**. Precision for "is a false positive" among flagged rows is
0.660 against a base rate of 0.465 — an 0.195 lift on a card the console prints
as a binary verdict.

More than a quarter of genuine planets are told the flux may come from a
neighbour. Alternatives, read off the confirmed-planet distribution:

| cut | flags of FPs | flags of confirmed |
|---|---:|---:|
| 3.0σ (served) | 61.8% | 27.6% |
| 6.20σ (p90 confirmed) | 46.0% | 10% |
| 10.43σ (p95 confirmed) | 36.8% | 5% |
| 30.75σ (p99 confirmed) | 26.0% | 1% |

## Result 3 — the scale is not sigma, and this is the finding that matters

| | share of all targets |
|---|---:|
| > 3σ | 43.49% |
| > 10σ | 20.26% |
| > 100σ | 7.96% |
| > 1000σ | **1.21%** |

Maximum **10,436σ**. A quantity genuinely denominated in standard deviations
does not reach four figures on one target in a hundred. `extract_centroid_features`
divides by `_robust_std` of the out-of-transit centroid and guards only
`sigma > 0`, so a target whose out-of-transit centroid barely moves — a quiet
star, a short baseline, a heavily-masked segment — produces an arbitrarily small
denominator and an unbounded ratio. The tail is a denominator artefact, not
astrophysics.

## Result 4 — how often it is not measured at all

**767 of 4,685 bulk-scored candidates (16.4%) have a NaN `centroid_snr`**, and
25 of 5,380 labelled rows. Until 2026-09-13 those were served as
`suspicious: false`, because `NaN > 3.0` is False, and the console rendered
them as a pass. That is [#19](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/19)
and it is fixed separately; it is recorded here because one candidate in six is
a large share of what this flag was reporting on.

## Why the threshold was not changed

Moving 3.0 to 10.43 would make the flag *look* reasonable — 5% of confirmed
planets instead of 27.6% — while leaving the reason it misbehaves untouched. It
would also be a number chosen because it produces an acceptable rate, which is
metric shopping under this project's own protocol (`CLAUDE.md` rule 6, and the
anti-falsification rules of #78): a cut picked off the outcome distribution it
is judged by, with no pre-registration and no held-out check.

The measurement says the denominator is broken. A threshold fitted on top of a
broken denominator inherits it, and the number would then have to be refitted
the moment the statistic is repaired. The order that survives scrutiny is:
repair `_robust_std`'s floor, re-measure, then set a cut against a
pre-registered criterion on held-out rows.

**Recommendation to the maintainer**, in the order the evidence supports:

1. Floor the denominator in `extract_centroid_features` — a scatter below the
   centroid's own quantisation is not a measurement — and re-measure this table.
2. Set the cut from that re-measurement, pre-registering the criterion first.
   "No more than 5% of confirmed planets flagged" is a defensible one and is
   what the p95 row above would give on today's statistic.
3. Until both are done, 3.0σ stays served and this file is the reason. It is
   already stated as a limit on the console's own diagnostics panel, which
   distinguishes unmeasured from passing.

Nothing was promoted and no served constant was changed by this work.
