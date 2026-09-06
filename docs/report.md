---
title: "Exoplanet Hunter — transit vetting with a calibrated dual-view CNN"
subtitle: "Served model `ca906040`, promoted 2026-07-19"
author: "Oliver Donaldson"
date: "2026-09-07"
geometry: margin=2.4cm
fontsize: 11pt
colorlinks: true
toc: true
toc-depth: 2
---

# Summary

This project vets transiting-planet candidates. It does not search for them: the
input is a signal someone has already flagged — a TESS Object of Interest, a
Kepler Object of Interest, a K2 candidate — with a published ephemeris, and the
output is a calibrated probability that the signal is a planet rather than an
eclipsing binary, a blend, or an instrumental artefact.

The served model is a dual-view 1-D CNN after Shallue & Vanderburg, trained with
5-fold cross-validation grouped by host star and calibrated per fold with Platt
scaling. Out of fold it reaches ROC-AUC **0.9581 ± 0.0057** and Brier
**0.0791 ± 0.0066** across five folds. On the mission that decides promotion,
TESS, it recovers **31.1%** of real planets at a 1% false-positive rate; on
Kepler, **81.3%**.

The larger part of the work was not the model that shipped but the eight
architectural candidates that did not. An eleven-branch ExoMiner-inspired
network was built, run, ablated, ensembled, extended twice, and closed. Every
one of those readings was pre-registered before its numbers existed and read
against a noise floor measured in the same run. None of them beat the model
above on the metric that governs, so none of them was promoted, and the record
of why is the substance of §4.

The single most important methodological commitment here is that **a margin
smaller than its own noise floor is not a result.** Applied consistently, that
rule closed a line of work this project had spent most of its compute on.

# 1. Problem formalisation

## 1.1 The task

Let a *candidate* be a periodic dimming signal in a star's light curve, given as
an ephemeris — period $P$, epoch $t_0$, duration $d$ — attached to a host star.
Vetting is the binary decision

$$y \in \{\text{planet}, \text{not a planet}\}$$

made from the star's photometry and its catalogue parameters. The model
estimates $\Pr(y = \text{planet} \mid \mathbf{x})$ where

$$\mathbf{x} = \big(\underbrace{\mathbf{g} \in \mathbb{R}^{2001}}_{\text{global view}},\ \underbrace{\boldsymbol{\ell} \in \mathbb{R}^{201}}_{\text{local view}},\ \underbrace{\mathbf{a} \in \mathbb{R}^{9}}_{\text{auxiliary}}\big)$$

These are the served model's actual input signatures: `global_view [None, 2001, 1]`,
`local_view [None, 201, 1]`, `aux [None, 9]`.

Framing it as vetting rather than detection is a real constraint, not a
simplification. The ephemeris is given, so the model never has to find the
period; equally, the model inherits whatever selection produced the candidate
list, which is the source of the principal limitation in §7.2.

## 1.2 Cost structure, and what follows from it

The two errors are not symmetric and neither is cheap.

A **false positive** consumes follow-up: radial-velocity time on an
oversubscribed spectrograph, or a night of ground-based photometry. A **false
negative** loses a planet — not permanently, but it drops off the shortlist that
anyone actually observes.

The scarce resource is follow-up capacity, and capacity is a *fixed budget*, not
a threshold on probability. An observer does not ask "is this above 0.5"; they
ask "I can afford 1% of my negatives as wasted nights — how many real planets
does that buy me?" That question is exactly **recall at a fixed false-positive
rate**, and it is the metric this project promotes on.

Two consequences run through the whole report:

1. **Recall @1% FPR is the decision metric.** ROC-AUC is reported because it is
   comparable to the literature, but a model can gain AUC while losing shortlist
   recall — stage 4's capacity arm did exactly that (§4, row 4) — and when the
   two disagree, recall governs.
2. **Calibration is a first-class metric, not a nicety.** A shortlist is a
   *ranking under a budget*. If scores are not comparable across the list, the
   cut is arbitrary. Expected calibration error (ECE) and the Brier score are
   therefore reported beside every accuracy number, and the promotion gate can
   reject a model for degrading either.

## 1.3 Why accuracy is not reported as a headline

Two independent reasons.

The labelled set is near-balanced **by construction** — 2,985 positive against
2,827 negative (§2.2) — because certain negatives are as hard to obtain as
confirmed planets. That balance is an artefact of labelling, not the deployment
prior.

And at deployment the prior is extreme in the other direction. Of the 3,919
TESS candidates the pipeline could actually process and score, **82.85%** score
at or above 0.5. A classifier that answered "planet" to everything would post an
accuracy in the eighties on that population while being useless for the decision
it exists to support. Accuracy measures the wrong thing here; the shortlist
metrics in §6 measure the right one.

# 2. Data collection

## 2.1 Sources

| Source | What it provides |
|---|---|
| NASA Exoplanet Archive — TOI, KOI (DR25), K2 tables | dispositions, ephemerides, stellar parameters |
| ExoFOP | community TOI/CTOI dispositions and comments |
| MAST, via `lightkurve` | SPOC and Kepler stitched light curves |
| Gaia DR3 | crossmatch for stellar parameters |

Provenance, per-table row counts and the sky coverage of each are in
[`data_provenance.md`](data_provenance.md). Raw FITS are treated as an evictable
cache of immutable archive files; everything derived is rebuilt by current code
and versioned in DVC (§8.2).

## 2.2 The labelled set

`data/tables/labels/labels.parquet` — 5,812 rows, one per host TIC, no
duplicates.

| Mission | Negative | Positive | Total |
|---|---:|---:|---:|
| TESS | 1,362 | 1,420 | 2,782 |
| Kepler | 1,250 | 1,250 | 2,500 |
| K2 | 215 | 315 | 530 |
| **All** | **2,827** | **2,985** | **5,812** |

Label rules:

| Disposition | Label |
|---|---|
| CP, KP (TESS); CONFIRMED (Kepler, K2) | 1 |
| FP, FA (TESS); FALSE POSITIVE, REFUTED (Kepler, K2) | 0 |
| PC, APC, CANDIDATE | held out — never trained on, never evaluated on |

Kepler negatives are restricted to DR25-certified false positives
(`koi_score < 0.5`), so a negative means *demonstrated not to be a planet*
rather than *not yet confirmed*. Without that restriction the negative class
would be contaminated with undiscovered planets and every recall figure in this
report would be optimistic by an unknown amount.

**The served model saw no K2 rows.** Of the 5,812 labelled rows, 4,818 carry an
out-of-fold prediction from the served run; joining those predictions to the
current catalogue gives TESS 2,371 and Kepler 2,237, **K2 zero**, with 210 rows
whose `tic_id` no longer joins a label row because the catalogue has been
refreshed since the run was trained. The 530 K2 rows were added to the labelled
set after `ca906040` was trained and have never been part of a promoted run's
population. Every K2 count in this section describes the data on disk today, not
the data the served model was fitted on, and no result in §6 is a K2 result.

## 2.3 The held-out candidate catalogue

`data/tables/catalogue/candidates.parquet` — 12,472 rows, TOI 8,148 and CTOI
4,324, spanning dispositions PC 4,828, FP 1,303, CP 816, KP 607, APC 485,
FA 102. This is the population the live console lists and the deployed model
scores. It overlaps the labelled set on the confirmed and false-positive rows;
the 5,104 PC/APC rows that are not in the labelled set are the genuinely
unresolved candidates the project exists to rank.

# 3. Preprocessing

The full specification is [`model_specs.md`](model_specs.md) §1; configuration
is `pipeline/conf/preprocess/default.yaml`. The steps, and why each is there:

1. **Stitch and normalise.** SPOC/Kepler PDCSAP flux, stitched across sectors
   or quarters, normalised per segment.
2. **Detrend with the transit masked.** A Savitzky-Golay filter removes stellar
   variability. The in-transit points are excluded from the fit, so the
   detrending cannot absorb the dip it exists to preserve — a filter fitted
   through the transit will happily flatten it.
3. **Phase-fold** at the catalogue ephemeris where one is published, from a BLS
   search otherwise.
4. **Median-bin** to two views: `global_view`, 2001 bins over the full phase
   $[-0.5, 0.5]$, and `local_view`, 201 bins over $\pm 3$ transit durations
   about phase zero. The global view carries the shape of the orbit and any
   secondary eclipse; the local view carries the transit's own profile at
   resolution the global view cannot afford.
5. **Auxiliary vector**, built by `features/aux.py::build_aux_row` — the single
   implementation shared by training and serving, so the two cannot drift.
   Imputed, log-transformed on the heavy-tailed columns, then standardised,
   with the transform fitted per fold and persisted in that fold's calibration
   bundle.

The served model consumes the 9-dimensional layout: `teff`, `radius`, `logg`,
`tmag`, `depth`, `duration`, `log_period`, `pink_snr`, `centroid_snr`. The
current build extends this to 13 with four scalar vetting diagnostics
(`oe_depth_sigma`, `oe_timing_sigma`, `secondary_sig`, `q_ratio`);
`LEGACY_AUX_DIM = 9` keeps older runs loadable bit-identically.

**A measured null worth keeping.** Adding those four features changed ROC-AUC by
$-1.3 \times 10^{-5}$ — indistinguishable from zero. The information was already
present in the folded views, and the features were *scalar summaries* of
diagnostics whose signal lives in shapes. That null is the origin of the branch
architecture in §4: feed the odd/even, secondary and centroid **views** rather
than statistics computed from them. It is also the first case in the project of
a plausible improvement being measured and found empty.

# 4. Model selection

## 4.1 How a candidate was allowed to win

Every architecture below was judged by the same procedure, in this order:

1. **Pre-register the reading.** What statistic, on what rows, against what
   comparator, and what result would falsify the hypothesis — written down and
   committed before the run produced numbers. The pre-registrations are files in
   [`experiments/`](experiments/README.md), each sitting immediately before the
   result it governs.
2. **Measure the noise floor in the same run.** With $n$ members trained per
   fold, the floor is $2\sigma/\sqrt{n}$ over members. A margin inside its floor
   is not a result.
3. **Run the promotion gate.** `promotion_gate.py` compares to the champion on
   out-of-fold TESS ROC-AUC, with Brier not degrading by more than 0.005 and ECE
   by more than 0.01, and recall @1% FPR not falling by more than the run's own
   measured floor. It returns one of three verdicts: **PROMOTE** (exit 0),
   **REJECT** (1), **UNRESOLVED** (2) — the last meaning the margin is inside
   the floor and the question is unanswered, which is a stop-and-ask rather than
   a rejection.
4. **Read the result exactly as pre-registered.** A result landing outside the
   terms fixed before it ran is recorded as falsified and never re-specified.

The gate has rejected several retrains. That is it working.

## 4.2 The record

Every row traces to a frozen file in [`experiments/`](experiments/README.md).
"Margin" is against the comparator in the same row; "floor" is the noise floor
measured in that run, by the rule above.

| # | Candidate | Decision metric | Measured | Comparator | Floor | Verdict |
|---|---|---|---:|---:|---:|---|
| 1 | Random forest on 14 hand-crafted features | — | — | — | — | **never scored** |
| 2 | **Dual-view CNN** (served) | TESS recall @1% FPR | **0.3069** | — | — | **champion** |
| 3 | Branch model, runs 1–3 | TESS recall @1% FPR | 0.238 / 0.126 / 0.145 | 0.307 | 0.034 | rejected |
| 4 | Branch + capacity | TESS recall @1% FPR | 0.236 | 0.307 | 0.034 | rejected |
| 5 | Leave-one-family-out attribution sweep | per-arm ΔAUC | null on every arm | — | — | no arm promoted |
| 6 | Control-arm host test (stage 7i) | host-AUC advantage | none measured | incumbent | — | criterion not met |
| 7 | Propensity weighting (stage 8) | baseline-dependence gap | **−0.1336** | 0 | 3.3× bar | bias fix, confirmed |
| 8 | Dual-view + branch ensemble (10.5) | shortlist recall | **0.4362 / 0.4223** | 0.3046 | — | complement, not promoted |
| 9 | Difference-image branch (stage 9) | control-arm host-AUC | unmeasurable | — | — | no effect shown |
| 10 | **Phase 1** — target-position channel | TESS recall @1% FPR, `dv_usable` | **−0.0254** | arm C'' | 0.0979 | **falsified (0.26×)** |

### Row 1 — the baseline this project does not have

`models/baseline_rf.py` and `conf/model/random_forest.yaml` are in the
repository, with a documented rationale, and the classical baseline is described
in the predecessor project's report. **No scored cross-validated result for it
exists anywhere in this repository**: `mlflow.db` holds 197 runs and every one
that records a model name records `cnn_dualview`. It is reported here as absent
rather than quietly omitted, and it is listed in [`known-limits.md`](known-limits.md)
as an outstanding gap. The honest reading is that the CNN's advantage over
classical ML is *assumed* in this project, not measured.

### Row 2 — the model that shipped

A dual-view 1-D CNN: two convolutional towers, one per view, concatenated with
the auxiliary vector and passed through a dense head, with Squeeze-and-Excitation
channel attention and residual late fusion. Full evaluation in §6.

Platt scaling rather than temperature scaling, for a measured reason: temperature
has no bias term and so cannot correct a distribution shift, which cost a 0.136
ECE regression before it was diagnosed and fixed.

### Rows 3–4 — the branch model, and why more capacity did not save it

The branch model is an ExoMiner-inspired rebuild: eleven views, one
convolutional branch each, reduced to per-branch embeddings and fused in a
shared head. Its structural argument over the dual-view design is
`unfolded_view` — up to 20 individual transits fed unstacked, so the network can
see whether a signal *recurs* rather than only the average of everything that
recurred.

Across three runs it scored 0.238, 0.126 and 0.145 TESS recall @1% FPR against
the champion's 0.307, on a decision floor of 0.034 — rejections at up to 4.8×
the floor, not close calls. Its resolution hypothesis was falsified separately
(Kepler moved +0.0707, the wrong direction for the proposed mechanism), and the
capacity arm closed the "it is simply too small" explanation: paired difference
−0.0035, $d = -0.44$. That arm is also the clearest case of AUC and shortlist
recall disagreeing — recall rose 0.145 → 0.236 while AUC fell.

### Row 7 — the one arm that changed something, and did not promote

Stage 8 tested propensity weighting against the observation-baseline confound
(§7.2). It worked: the branch model's amplification of baseline dependence was
removed, a gap of −0.1336 at 3.3× its bar, at no cost in recall. It is recorded
as a **bias fix**, not a promotion, because it does not improve the decision
metric — it removes a reason to distrust the model's ranking. All four of the
stage's own predictions were falsified; the win it did produce was not one it
had predicted.

### Row 8 — the complement finding

Ensembling the dual-view and branch models reached shortlist recall 0.4362 and
0.4223 against the common-fold dual-view member's 0.3046. That is a real gain,
and it is the strongest argument in the record that the branch line was learning
*something* the dual-view model was not.

It did not promote, for a reason worth stating plainly: an ensemble of two
architectures is not a candidate for the champion slot under this project's
serving constraints — it doubles inference cost on a scale-to-zero deployment,
and the gate compares single models to the champion. It is recorded as a
complement finding: evidence that the branch's information is not redundant,
banked for a future design rather than shipped.

### Rows 9–10 — closing the line

Stage 9 added a difference-image branch — pixel-level source location, the
strongest single nearby-eclipsing-binary discriminant. Its falsification test was
**unmeasurable** as specified, because the control-arm harness zeroes the branch
on exactly the rows the test needed. Three secondary predictions confirmed it
cost nothing; none showed it delivered anything.

Stage 9's readout named one remaining explanation for that null: the stamps were
fed without the star's own pixel position, so the network had no reference frame
in which a centroid shift means anything. Phase 1 built that channel and ran the
paired arms C'' (branch dropped at the model) and D'' (branch kept), on identical
augmented batches.

On the pre-registered statistic — TESS recall @1% FPR on `dv_usable` rows,
$n = 2{,}077$, 1,220 positive:

| | arm C'' | arm D'' | D − C | floor | ratio | verdict |
|---|---:|---:|---:|---:|---:|---|
| recall @1% FPR | 0.2197 | 0.1943 | **−0.0254** | 0.0979 | 0.26× | within floor |
| ROC-AUC | 0.9228 | 0.9188 | −0.0040 | 0.0154 | 0.26× | within floor |

Two floors could have governed — a member-pairing floor and the Phase 1a seed
floor of 0.0432 — and which one would be used was fixed in writing at 03:52
while arm D'' was still training, with an explicit rule that disagreement between
them would be recorded UNRESOLVED. They agreed; the margin is inside both.

**Read exactly as pre-registered, the reference-frame explanation is falsified.**
No third stamp variant was commissioned and the branch line is closed in
[`decisions.md`](decisions.md). Two things are stated because omitting either
would flatter the result: the margin is negative in every cell, and it is not
large enough to be called harm either — the honest reading is *no effect*. And
one post-hoc number is recorded and explicitly not banked: TESS ECE improved
0.0342 → 0.0238 from C'' to D'', with no ECE floor pre-registered.

## 4.3 What the record shows

Ten candidates; one served; one bias fix adopted; one complement finding banked
but not shipped; one line of work closed by its own pre-registered criterion.
The compute spent on rows 3–10 bought no improvement to the served model. It
bought something else: a documented reason to believe the served model is not
being beaten by an obvious alternative, and a demonstration that this project's
gate cannot be talked past.

# 5. Training

## 5.1 Cross-validation

**5-fold `StratifiedGroupKFold`, grouped by host star, stratified on label.**
The grouping is the correctness-critical part: multi-planet systems and
re-observed targets contribute several rows that share a star, and a star split
across folds leaks. In the predecessor project, moving from a single 70/15/15
split to grouped k-fold cost 2–5 AUC points — that drop is the leakage being
removed, not a regression.

Within each outer fold an inner 88/12 `GroupShuffleSplit` separates training from
validation. The inner validation set drives three things and is never used for
reported metrics: early stopping, the F1-optimal decision-threshold sweep, and
the calibrator fit.

## 5.2 The served run

Champion `ca906040cdb74ba6b07353a500244777`, promoted 2026-07-19, unchanged
since. Its five folds are MLflow child runs of that parent:

| Fold | Epochs recorded |
|---|---:|
| fold-0 | none — no per-epoch series was logged |
| fold-1 | 78 |
| fold-2 | 79 |
| fold-3 | 70 |
| fold-4 | 84 |

Early stopping on inner-validation loss accounts for the spread. Fold-0's
missing series is a logging gap, not a training failure: its weights, its
out-of-fold predictions and its calibration bundle are all present and it
contributes to every pooled number in §6. The training-curve figure names the
missing fold in its title rather than silently drawing four lines and calling
them five.

## 5.3 Calibration and uncertainty

Platt scaling is fitted per fold on the inner validation logits: the fold's
mean intercept is $b = 0.942 \pm 0.554$ and slope $a = 0.916 \pm 0.087$. The
fitted decision threshold is $0.486 \pm 0.039$ — near 0.5, which is what
near-balanced labelled data should give, and reported rather than assumed.

At serving, uncertainty comes from Monte-Carlo dropout: repeated stochastic
forward passes give `prob_mean`, `prob_std` and a p10–p90 interval per candidate,
all surfaced on the console. Fold disagreement is reported separately from MC
spread, because they measure different things — the console labels them
`Total sigma (MC + fold)` and `Within-fold sigma` rather than showing one number twice.

## 5.4 Ensembling, and the floor

Both trainers accept `n_models_per_fold`. Members are trained independently,
their scores averaged, and the calibrator fitted once over the average; each
member's uncalibrated score is written out, so the spread the run averaged over
stays readable afterwards. **That spread is the noise floor every margin in this
project is read against.**

The served model trains **one member per fold**, so it has no seed spread of its
own and therefore **no measured noise floor**. The API reports this in exactly
those words rather than substituting a floor measured on a different
architecture — a branch-model floor printed under dual-view numbers would be a
category error, and fixing an instance of it was a deliberate step of this work.

# 6. Evaluation

## 6.1 Protocol

Every number in this section is **out of fold**: each row is scored by the fold
that did not train on it. There is no additional held-out test set, and this is
a deliberate trade — with 5,812 labelled rows, a 15% holdout would be ~870 rows,
too few to resolve a per-mission recall difference at the floors this project
reads against. The cost is that no number here is protected from the model
selection in §4, and the mitigation is the pre-registration discipline: readings
were fixed before results existed.

Pooled predictions: 4,818 rows, 2,551 positive.

## 6.2 Cross-validated performance

Fold means ± standard deviation across the five folds, from
`models/cv/ca906040.../cv_summary.json`:

| Metric | Mean | SD |
|---|---:|---:|
| ROC-AUC | 0.9581 | 0.0057 |
| PR-AUC | 0.9599 | 0.0048 |
| F1 (at the fitted threshold) | 0.9001 | 0.0125 |
| Brier | 0.0791 | 0.0066 |
| ECE | 0.0276 | 0.0048 |
| Decision threshold | 0.486 | 0.0388 |

## 6.3 Per mission, at the operating point

TESS decides promotion; Kepler is diagnostic. K2 does not appear because the
served run contains no K2 rows (§2.2). Each mission's confusion
matrix is cut at **its own 1% FPR threshold**, and the recall printed beside it
is the recall at that same cut — one measurement, not two.

| | **TESS** (gating) | **Kepler** (diagnostic) |
|---|---:|---:|
| n / positive | 2,371 / 1,304 | 2,237 / 1,245 |
| ROC-AUC | 0.9100 ± 0.0088 | 0.9914 ± 0.0028 |
| Recall @1% FPR | **0.3113 ± 0.0656** | 0.8129 ± 0.0511 |
| Brier | 0.1210 ± 0.0092 | 0.0360 ± 0.0055 |
| ECE | 0.0435 ± 0.0134 | 0.0407 ± 0.0039 |
| Threshold at 1% FPR | 0.9612 | 0.8314 |
| Actual FPR at that cut | 0.0103 | 0.0101 |
| TP / FP / FN / TN | 406 / 11 / 898 / 1,056 | 1,012 / 10 / 233 / 982 |
| Precision at that cut | 0.9736 ± 0.0066 | 0.9902 ± 0.0025 |
| F1 at that cut | 0.4718 ± 0.0734 | 0.8928 ± 0.0306 |

Recall @5% and @10% FPR, on the gate's re-scored population (see the note
below): TESS 0.5608 and 0.7308; Kepler 0.9575 and 0.9888.

**Two populations, one run.** The record and the promotion gate quote TESS
recall @1% FPR as **0.3069** on 2,367 rows, from
`models/cv/champion-rebaselined-today/cv_summary.json` — the champion re-scored
out of fold on the labelled set as it stood on 2026-08-17. The API computes
**0.3113** on 2,371 rows, joining the same predictions to the labelled catalogue
as it stands today. Same model, same predictions, four rows of difference in the
join. Both are reported here; §4's table uses the gate's number because that is
the number promotion decisions were made against.

## 6.4 Reading those numbers

**The Kepler–TESS gap is the headline finding of the evaluation.** ROC-AUC 0.991
against 0.910, and recall @1% FPR 0.81 against 0.31, on the same model at the
same time. Kepler stared at one field for four years; TESS mostly gets 27 days
per sector. The transits are shallower, the baselines shorter, the systematics
worse. A model reported only on its pooled AUC of 0.958 would look far more
capable than it is on the mission that actually matters for new discoveries.
This is why TESS is the gating mission and why per-mission slicing is not
optional.

**Precision is high and F1 is low on TESS, and that is expected.** At a 1% FPR
cut the model makes 417 positive calls of which 406 are right — but there are
1,304 real planets, so it misses 898. F1 balances precision and recall as though
they were equally valuable; under a fixed follow-up budget they are not. F1 is
reported for completeness and is not a promotion criterion.

**Calibration is good pooled and mediocre per mission.** Pooled ECE is 0.0121;
per mission it is 0.0435 (TESS) and 0.0407 (Kepler). Pooling cancels errors of
opposite sign — TESS is over-confident where Kepler is under-confident — so the
pooled figure flatters. The per-mission figures are the ones to read.

## 6.5 Figures

All regenerated from the served run by
`pipeline/scripts/make_performance_figures.py`.

![Out-of-fold ROC by mission, drawn at unit aspect with the 1% FPR shortlist
point marked. The marked values, 0.307 on TESS and 0.798 on Kepler, use the
conservative cut — the highest recall reachable *without* exceeding the budget —
which is the convention the promotion gate uses (§6.3).](figures/roc_operating_point.png)

![Reliability diagram: predicted probability against observed frequency, ten
bins, pooled out of fold.](figures/calibration.png)

| Figure | What it shows |
|---|---|
| `figures/roc_operating_point.png` | Per-mission ROC at unit aspect, 1% FPR operating point marked |
| `figures/roc_pr.png` | Pooled ROC and precision–recall curves |
| `figures/calibration.png` | Reliability diagram, 10 bins, with the diagonal |
| `figures/training_curves.png` | Per-fold loss and AUC by epoch; the title names the fold with no series |
| `figures/embedding_3d.png` | Penultimate-layer embedding, coloured by label |
| `figures/risk_coverage.png` | Risk against coverage as the score threshold sweeps |
| `figures/completeness.png` | Recovery against transit depth and period |
| `figures/sky_map.png`, `figures/coverage_map.png` | Sky and sector coverage of the labelled set |

# 7. Results and limitations

## 7.1 What the deployed system produces

The API scores the held-out candidate catalogue in bulk and serves the results
to the console. On the most recent bulk pass, of 4,685 candidates attempted:

| Outcome | Count |
|---|---:|
| Scored | 3,919 |
| No light curve available at MAST | 744 |
| Preprocessing failed | 22 |

Of the 3,919 scored, **82.85%** score at or above 0.5 and **1.28%** at or above
0.9. That distribution is itself a result: the candidate list is dominated by
signals that already look planet-like, which is why a 0.5 cut is nearly
vacuous on this population and why the console's default ranking is by score
under an explicit budget rather than by a fixed threshold.

## 7.2 Limitations

**The observation-baseline confound (W1).** How long a star was watched
correlates with its label: on TESS the correlation between observation baseline
and label is +0.387. Longer baselines produce more confirmations, so a model can
gain apparent skill by learning observing strategy rather than astrophysics. It
is measured, not hypothesised: the branch model amplified the correlation to
+0.5155, which is 0.13 *above* the labels' own. Stage 8 showed propensity
weighting removes the amplification (§4, row 7). The confound in the labels
themselves remains, and every recall figure in this report is read under it.
The console now shows the derived baseline on each candidate row so a reader can
see the confound rather than take it on trust.

**The model scores the star, not only the signal (W2).** Some of the model's
discrimination comes from host properties rather than from the transit. The
control-arm harness that would quantify this zeroes the branch on exactly the
rows that would test it, which is why stage 9's primary criterion was
unmeasurable. A control-arm host-AUC floor was measured for the first time in
Phase 1a — 0.0198, from three seeds at mean 0.5826 — so the instrument now
exists even though the stage-9 question stayed unanswered.

**No classical baseline.** §4, row 1. The CNN's advantage over hand-crafted
features is assumed here, not measured.

**No measured noise floor for the served model.** One member per fold (§5.4).
Every floor quoted in this report was measured on a different run, and is used
only to read that run's own margins.

**The unexplained Kepler cell (W7).** A +0.1446 movement in one Kepler cell has
never been explained. It is recorded rather than resolved.

**Unbounded cross-validation runtime (W10).** Fold duration within a single
Phase 1 arm ranged from 25 to 78 minutes with no bound on the slowest fold,
which makes compute budgeting unreliable for any future sweep.

**Evaluation is out of fold, not on a fresh holdout.** §6.1.

The full register, with each item's evidence and status, is
[`known-limits.md`](known-limits.md).

# 8. Reproducibility

## 8.1 Environment and commands

```bash
make env      # conda env from environment.yml, editable installs
make test     # fast suite (network and slow markers excluded)
make lint     # ruff
make type     # mypy against the tracked baseline
make validate # data validation gates on whatever artefacts exist
make refresh  # the full refresh DAG; trains only if warranted
make report   # render docs/report.md to docs/report.pdf
```

## 8.2 Data and model versioning

Artefacts are DVC-tracked with a Cloudflare R2 remote: `.dvc` pointer files are
in git, the bytes are in R2. `make data-pull` materialises them; `make data-push`
syncs. The promoted model is pinned in `models/registry.json`, which carries the
run id, its `cv_summary.json` path and the metrics it was promoted on. That file
is not edited by hand — only the promotion gate writes it.

## 8.3 Experiment tracking

MLflow (`mlflow.db`, `mlruns/`), one parent run per cross-validation with a child
run per fold. Every number in §6 comes from either `cv_summary.json` or
`predictions.parquet` under `models/cv/<run_id>/`.

## 8.4 Serving

FastAPI on Fly.io, scale-to-zero, with the console a single static build on
Render that deploys from `main`. The API deploys manually:

```bash
fly deploy --remote-only
```

The `/score/{tic_id}` contract is pinned: `api/app/schemas.py` and the console's
client `frontend/design-console/src/app.api.js` change together or not at all,
and a contract test extracts every field the client reads and asserts it exists
on the Pydantic model, so the two cannot drift silently.

## 8.5 The record

[`experiments/`](experiments/README.md) holds one frozen file per stage or
audit, in the order things happened, with each pre-registration immediately
before the result it governs. Nothing there is rewritten: a correction is a
dated note appended under the entry it corrects. Every number in §4 traces to
one of those files.

# 9. References

Full bibliography in [`references.bib`](references.bib).

| Key | Why it is cited here |
|---|---|
| `shallue2018` | the dual-view architecture the served model follows |
| `valizadegan2022`, `valizadegan2025` | ExoMiner and ExoMiner++ — the per-diagnostic branch design §4 is *inspired by*, not a reimplementation of |
| `twicken2018`, `thompson2018` | Kepler DR25: the certified false-positive population used for the negative class |
| `jenkins2016` | the SPOC pipeline that produces the light curves |
| `giacalone2021` | the vetting problem as the community frames it |
| `lightkurve2018` | the library the ingest layer uses |
| `platt1999` | Platt scaling, the calibrator fitted per fold |
| `gal2016` | MC dropout, the serving uncertainty estimate |
| `nasa`, `exofop` | the catalogues the labels come from |
