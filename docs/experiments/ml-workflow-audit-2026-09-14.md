# End-to-end audit against the taught workflow · 2026-09-14

Part **A** of [#78](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/78):
*"Audit the existing pipeline against the ML workflow taught in the maintainer's
coursework… the session must read it and cite which standard each finding is
measured against, rather than applying generic best practice."*

Every finding below was produced by running something. Nothing here is a reading
of the code alone, and where a check was inconclusive it is recorded as
inconclusive.

## 0. Which standard, and one correction before starting

The course material supplied is: Frery, *Machine Learning* I–V; DATA 305/475
lectures L1–L20; DATA 303/473 *Lecture Notes Weeks 1–6* and its weekly slides;
Marsland (2014); Géron, *Hands-On Machine Learning*, 2nd ed.

| audit section | standard cited |
|---|---|
| workflow spine | Géron, Appendix B, *Machine Learning Project Checklist* |
| preprocessing | DATA 305 **L6 Data Pipelines & Preprocessing** |
| training | DATA 305 **L4/L5 Training Deep Networks I/II**, **L3 Keras API** |
| splitting, test error | DATA 303 **§7.1 Assessing model accuracy** |
| evaluation metrics | Frery **ML II, Performance metrics**; DATA 303 **Week 9** |
| classical baseline | Marsland **Ch. 12–13**; course notes *Decision Trees*, *Random Forests* |

**Correction: the workflow this repository claims to follow is not the one the
supplied material teaches.** `PLAN.md` and #78 both describe a seven-step
workflow — *"formalise the problem, collect data, preprocess, select a model,
train, evaluate, report"* — attributed to DATA 301. The DATA 301 material on
disk contains no such statement. Frery's ML I teaches a **five**-step flowchart
(dataset collection → preprocessing → model selection → training → evaluation),
and Géron's Appendix B an **eight**-step checklist. The repository's seven steps
are closest to Géron's with *Explore the data* and *Launch, monitor and maintain*
dropped.

The spine used here is **Géron's eight**, because it is the only supplied source
that states one in full, and because the two steps the repository dropped are the
two this audit has the most to say about.

## 1. Frame the problem — Géron B.1

Géron B.1 asks, as items 5 and 6, *"How should performance be measured?"* and
*"Is the performance measure aligned with the business objective?"*

**Failed, and separately measured.** The gating statistic was TESS recall @1% FPR,
which [P2.1](p2-1-power-analysis-2026-09-14.md) shows has an MDE of 0.1222 at 5
members per fold and cannot detect a change confined to the follow-up region —
the region the product exists to order. The metric was not aligned with the
objective and, worse, could not detect misalignment. Closed by P2.1; the gate now
reads ROC-AUC with a pAUC veto.

## 2. Get the data — Géron B.2

Measured on `labels.parquet` after the 2026-09-13 refresh, and on the served run's
`predictions.parquet`.

| check | result |
|---|---|
| rows / distinct `tic_id` | **5,826 / 5,826** — max group size **1** |
| class balance | TESS 0.509 pos, Kepler 0.500, K2 0.594 |
| Kepler rows | exactly **1,250 / 1,250** |
| K2 rows in the served run | **0 of 530** |

**The Kepler cap is confirmed and its mechanism is explicit.**
`conf/data/full.yaml` sets `n_confirmed_kepler: 1250` and `n_false_pos_kepler:
1250`; `catalog.py::_stable_sample` md5-ranks by `seed:tic_id` and takes the
first *n*. Measured eligible pool, from the refresh's own log on 2026-09-13:
**KOI confirmed = 1,945, FP = 3,719**. So 2,500 of 5,664 eligible rows are used —
**44.1%**, and 3,164 rows are discarded.

> **Correction to `known-limits.md`.** The carried limit states *"roughly 2,748
> eligible confirmed and 3,813 eligible FPs"*. The measured pool today is 1,945
> and 3,719. The FP figure is close; the confirmed figure is out by 803 rows,
> about 41%. The "45% subsample" headline survives — it is 44.1% on today's
> numbers — but the component counts do not.

**K2 is not capped.** `full.yaml` requests 1,000,000 of each and the pool is 530.
The served run saw none of them because it predates their addition, which is a
provenance fact and not a sampling decision.

**Unit of analysis (Géron B.2.10, "check the size and type of data").** One row
per host, so the TCE is not the unit and every multi-planet system loses its extra
rows. This is confirmed independently below, because it is also what makes the
splitter inert.

## 3. Prepare the data — Géron B.4, DATA 305 L6

L6 is *Data Pipelines & Preprocessing*. Each step was checked on a case with a
known answer, as #78 requires.

| # | check | result |
|---|---|---|
| P1 | phase folding centres a known transit | **pass** (after correcting the check — see below) |
| P2 | `_normalise` puts baseline at 0, deepest bin at −1 | **pass**, median 0.00e+00, min −1.000000 |
| P3 | normalisation is depth-invariant (10× depth → same view) | **pass**, max abs difference 0.00e+00 |
| P4 | `bin_profile(n)` returns exactly *n* bins | **pass** at 2001 and 201 |
| P5 | folding at the wrong period destroys the dip | **pass**, depth 0.01000 → 0.00000 |
| P6 | the local window contains in-transit cadences | **pass** |

*P1's first run reported a failure at 116 bins off centre. That was the check, not
the pipeline: the synthetic transit is flat-bottomed, so `argmin` returns the
first bin of the floor rather than its centre. Recorded because an audit that
reports its own false positives as findings is worth less than one that does not.*

### 3.1 The finding: the fold window is specified in phase and applied in days

`build_views` computes the local half-width as `local_durations * duration /
period` and comments it *"half-window in phase units"*. It reaches
`bin_profile` as a bound on `lc.fold(...).time`, and **lightkurve returns folded
time in the light curve's own units — days.** Verified by execution: for P = 3.5 d
the folded axis spans [−1.7500, +1.7480], not [−0.5, +0.5].

Both windows are therefore in days:

- **the global view spans ±0.5 days**, which is 1/P of the phase. Measured through
  the real `build_views`: 100% of phase at P = 0.8 d, **28.6%** at P = 3.5,
  **10.0%** at P = 10, **3.3%** at P = 30;
- **the local view spans ±3D/P days**, which is **3/P durations**, not 3. At
  P = 10 d it is 0.30 durations; at P = 30 d, 0.10 — entirely inside the transit,
  so the view holds no out-of-transit baseline at all and `_normalise` amplifies
  whatever noise is left.

**This contradicts the console.** The About page tells visitors the global view is
*"2,001 bins across the whole phase"* carrying *"orbital shape and any secondary
eclipse"*. A ±0.5 day window cannot reach phase 0.5 for any period above 1 day, so
for 4,595 of the 5,826 labelled targets the global view cannot contain a secondary
eclipse. Verified by building a light curve with a secondary and confirming no dip
appears near the view edges.

**Evidence that the shipped views carry it.** The local view's baseline fraction —
the share of bins above −0.2 — correlates with log period at **+0.4351** over
5,321 targets. A window fixed at ±3 durations is period-independent by
construction and produces a flat ≈0.83 at every period; it cannot generate a
trend. A ±3D/P day window can, and does:

| period | shipped | synthetic, this code path, 1000 ppm noise |
|---|---:|---:|
| 0–2 d | 0.766 | 0.826 |
| 2–5 d | 0.667 | 0.826 |
| 5–10 d | 0.726 | 0.896 |
| 10–30 d | 0.811 | 0.960 |
| 30–100 d | 0.940 | 0.975 |
| 100+ d | 0.990 | 0.985 |

The direction and range agree; the mid-period rows sit lower in the shipped data
than in the synthetic. **The trend's existence is the finding; the exact profile
is not reproduced, and no claim is made that it is.** An earlier pass predicted
the opposite signature — baseline vanishing rather than saturating — and was
wrong, because an all-in-transit window is flat and `_normalise` returns its
`depth < 1e-8` branch rather than scaling a dip to −1. That mistake is recorded
because it nearly produced a "no bug here" conclusion.

Captured as `pipeline/tests/test_view_window_units.py`: three strict `xfail`s
asserting the intended behaviour, so they flip to failures the moment the units
are reconciled and the markers must be removed.

**Not fixed here, deliberately.** Reconciling the units changes every model input
and invalidates the served run's training set. That is a decision for the
maintainer and belongs with the clean retrain (#78 part B), not inside an audit.

## 4. Splitting and leakage — DATA 303 §7.1

§7.1 sets the standard: the quantity that matters is **test** error, estimated on
data the model did not see.

**Confirmed inert.** `labels.parquet` is 5,826 rows on 5,826 distinct `tic_id`,
so `StratifiedGroupKFold` has max group size 1 and partitions identically to
`StratifiedKFold`. The project is leak-free through the label builder's
one-row-per-host emit, not through the splitter.

**The consequence for the test suite is worse than the register says.** Every
existing grouping test is vacuous — it passes whether or not grouping works,
because there is nothing to group. `pipeline/tests/test_leakage_probe.py` builds a
TCE-level set (150 rows, 50 hosts, 3 each) and applies one shared `straddling`
check to the real splitter and to an ungrouped one.

**The probe was mutation-tested against the real implementation**, which #78
requires ("a leakage probe that demonstrably fails when host grouping is broken"):

| mutation | result |
|---|---|
| `build_fold_assignment` stripped of grouping | probe fires |
| `stratified_inner_split` stripped of grouping | **`test_the_inner_split_keeps_a_host_whole` fails, 27 straddling hosts** |

`splits.py` was restored after each; `git diff` clean.

**The mutation also caught a defect in the first version of the probe.** A
map-level check I had written was vacuous for the same reason the existing tests
are: `build_fold_assignment` returns a dict keyed by host, so a straddling host is
unrepresentable and the check could never fail. It was replaced by the fold-level
assertion, which can.

## 5. Evaluation — Frery ML II, DATA 303 Week 9

Metric definitions are consistent between the analysis harness and the served
API: `power_analysis.py::recall_at_fpr` is pinned by test against
`app.routes.model::_recall_at_fpr`, imported rather than copied.

The substantive evaluation findings are in
[P2.1](p2-1-power-analysis-2026-09-14.md) and are not repeated here. The one that
belongs in this section: **seed sd exceeds sampling sd for every candidate
metric**, so an interval built from a paired bootstrap alone measures the smaller
half of the uncertainty.

## 6. Tests — what they assert versus what they would catch

#78 names this directly: *"A passing suite that cannot fail on a real defect is
the failure mode this repo exists to avoid."*

Two instances found, both by mutation rather than by reading:

1. **The grouping tests cannot fail** (§4). Max group size is 1, so the guarantee
   they assert holds trivially.
2. **My own first leakage probe could not fail** (§4). Written by someone
   deliberately looking for this failure mode, and it still happened.

The general lesson is the one Géron B.4 implies and this repository states in rule
8: a check is worth its runtime only if something can make it go red. **No
mutation testing exists in this project**, and the two defects above were found in
the first hour of looking. That is a recommendation, not a measurement: the 898
fast tests were not individually assessed, and this audit does not claim a rate.

## 7. What this audit did not cover

Stated rather than quietly omitted:

- **Detrending** (`clean.py::flatten_lightcurve`) was not verified on a known
  answer. It is the step most likely to hold another units-style defect.
- **The augmentation RNG's statefulness** is carried in `known-limits.md` and was
  not re-measured here.
- **Training**: the augmentation-is-training-only guarantee, what early stopping
  monitors, and W10's unbounded slowdown were not exercised.
- **Calibration**: the Platt fit's per-fold behaviour was not checked.
- **The 898 fast tests** were not individually assessed for whether they can fail.
- **The classical baseline** (Marsland Ch. 12–13) is P2.3 and is not started.

## 8. What follows

1. **The window units are the largest open defect** and they gate the clean
   retrain: retraining on inputs built by this code would bake the same windows
   into a new run. Decide before P2.6.
2. **The console's "whole phase" and "secondary eclipse" claims are wrong today**
   regardless of what is decided about the code, and that is visitor-facing.
3. `known-limits.md`'s Kepler eligibility counts need the correction in §2.
4. Mutation testing deserves a place in the suite.

Nothing was promoted, nothing was trained, `models/registry.json` is untouched.
