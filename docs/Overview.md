# Exoplanet Hunter Pipeline Outputs

For pipeline version 2.

## 1. Introduction

Exoplanet Hunter is an end-to-end pipeline for the vetting of transiting planet
candidates in NASA's TESS, Kepler and K2 data. Given a target and an ephemeris
it returns a calibrated probability that the signal is a planet, together with
the diagnostic evidence behind that probability.

The pipeline is designed for triage. A space telescope produces far more
candidates than can be examined by hand, and most periodic brightness dips are
not planets — they are eclipsing binaries, background eclipses bleeding into the
aperture, instrumental systematics, or stellar variability. Separating these is
called vetting, and it is the bottleneck. The purpose of the pipeline is to rank
a candidate queue so that human attention goes where it pays, and to attach a
probability that can be acted on rather than a bare yes or no.

The pipeline generates three products: a **prediction table** carrying scores
for every evaluated target, a **cross-validation summary** recording how the
model that produced them was measured, and a **scoring response** served over
HTTP, which carries the diagnostic panels a human vetter reads. This document
describes those outputs. It does not describe how to run the pipeline — see
[getting-started.md](getting-started.md) — nor how the model is built, which is
[model_pipeline.md](model_pipeline.md) and [model_specs.md](model_specs.md).

## 2. Prediction Table

The prediction table is the primary product of a cross-validation run. It is
written as `predictions.parquet` in the run directory and holds one row per
evaluated target, scored **out of fold** — every row is predicted by a model
that did not train on it.

### 2.1 Identity and label columns

| Column | Description |
|---|---|
| `tic_id` | TIC identifier of the target (integer scalar). K2 targets carry their EPIC identifier in this column. |
| `mission` | Source mission: `TESS`, `Kepler` or `K2` (string). |
| `label` | Ground truth: `1` planet, `0` false positive. Candidates are held out of training and do not appear. |
| `fold` | Cross-validation fold that held this target out (integer, 0–4). |

### 2.2 Score columns

| Column | Description |
|---|---|
| `score` | Calibrated planet probability in [0, 1]. This is the served number. |
| `member_score_N` | Uncalibrated score from ensemble member `N`. Present once a run trains more than one model per fold; the spread across members is the run's reseeding noise. |

Members are averaged and calibrated **once over the average**, not calibrated
individually and averaged afterwards: the ensemble is what is served, so the
Platt fit has to describe it rather than any one member.

### 2.3 Ephemeris and observation columns

| Column | Description |
|---|---|
| `period` | Orbital period in days (float scalar). |
| `duration` | Transit duration in days (float scalar). |
| `depth` | Transit depth as a fraction of stellar flux. |
| `observed_transit_count` | Transits actually present in the light curve. |
| `expected_transit_count` | Transits predicted over the observation baseline. |
| `transit_completeness` | Observed over expected. |
| `n_sectors_observed` | Number of TESS sectors covering the target. |
| `lc_source` | Archive the light curve was drawn from. |

`expected_transit_count` and `period` together give the **observation baseline**
in days, the quantity the pipeline's own bias measurements are computed against.
See [data_provenance.md](data_provenance.md).

### 2.4 Data Validation columns

Where the mission pipeline publishes a Data Validation report, its diagnostic
statistics are carried through unmodified. These are inputs to the branch models
and are recorded so that any score can be traced to the evidence available when
it was produced.

| Group | Columns |
|---|---|
| Detection strength | `max_multiple_event_sigma`, `max_single_event_sigma`, `max_ses_in_mes`, `model_fit_snr`, `robust_statistic` |
| Statistical significance | `bootstrap_significance`, `bootstrap_threshold_pfa`, `chi_square_gof`, `chi_square_gof_dof` |
| Odd–even | `odd_even_statistic`, `odd_even_significance` |
| Weak secondary | `weak_secondary_max_mes`, `weak_secondary_depth_ppm`, `weak_secondary_robust_statistic`, `albedo_comparison_statistic` |
| Ghost diagnostic | `ghost_core_statistic`, `ghost_core_significance`, `ghost_halo_statistic`, `ghost_halo_significance` |
| Centroid | `mean_sky_offset`, `mean_sky_offset_uncertainty`, `control_sky_offset`, `control_sky_offset_uncertainty` |
| Period aliasing | `longer_period_statistic`, `shorter_period_statistic`, `matched_period_days`, `period_mismatch_frac` |
| Difference imaging | `n_difference_images`, `diff_image_min_px`, `diff_image_max_px`, `diff_quality_median` |
| Stellar parameters | `effective_temp`, `log_g`, `log_metallicity`, `stellar_density`, `stellar_radius`, `stellar_mass`, `tess_mag` |
| Astrometry | `ruwe`, `has_ruwe`, `n_dr3_candidates` |
| Coverage | `dv_usable`, `summary_quality_fraction`, `dv_observed_transit_count`, `dv_expected_transit_count` |

`dv_usable` is `False` where no Data Validation report was available. The
remaining columns in that row are then null, and are imputed downstream rather
than dropped, so a target is never silently excluded for missing diagnostics.

## 3. Cross-Validation Summary

Each run writes `cv_summary.json` beside its prediction table. It records what
the run measured and the configuration that produced it, so two runs can be
compared without reading either one's code.

### 3.1 Headline metrics

`summary` carries mean and standard deviation across folds for `test_roc_auc`,
`test_pr_auc`, `test_f1`, `test_brier` and `test_ece`.

`per_mission` repeats those per mission and adds the operating points the
shortlist is read at — `recall_at_1pct_fpr`, `recall_at_5pct_fpr`,
`recall_at_10pct_fpr` — with `n` and `n_positive`, for each of `TESS`, `Kepler`,
`K2` and `all`. Mission separation is not cosmetic: the missions differ by
several points of AUC, and a pooled figure hides it.

### 3.2 Variance block

A margin cannot be read without the noise it sits against, so every run reports
its own.

| Field | Description |
|---|---|
| `fold_sd` | Spread of AUC across folds — fold difficulty. |
| `seed_sd` | Spread of AUC across members within a fold — reseeding noise. |
| `recall_fold_sd`, `recall_seed_sd` | The same decomposition, for recall. |
| `gate_recall_fold_sd`, `gate_recall_seed_sd` | The same, on the TESS gating slice the shortlist is drawn from. |
| `pooled_gate_recall` | Gate recall for each member's complete out-of-fold set. |
| `pooled_gate_recall_seed_sd` | Spread of those pooled draws — the run-level reseeding noise, directly. |
| `n_models_per_fold` | Members trained per fold. |

The decision rule is `2 x sd / sqrt(n_models_per_fold)`. **A margin smaller than
that is not a result.** The rule has correctly rejected several of this
project's own changes.

### 3.3 Run configuration

`run_config` records `n_splits`, `val_frac`, `epochs`, `batch_size`, `patience`,
`learning_rate`, `seed`, `n_models_per_fold`, the augmentation settings, any
`baseline_intervention`, the `fold_assignment` file when the outer split was
pinned, `view_shapes`, `n_examples`, the full `model_config`, and the git
provenance of the code that ran (`git_sha`, `git_dirty`, `git_branch`).

Two runs differing only in their fold assignment are distinguishable from their
summaries alone. This matters because comparing models trained on different
splits compares populations rather than models.

## 4. Scoring Response

`GET /score/{tic_id}` downloads the light curve, runs the full preprocessing
path and the ensemble, and returns the result as JSON. It is the same code path
that produced the prediction table, so a served score and a recorded score are
the same quantity.

### 4.1 Headline fields

| Field | Description |
|---|---|
| `tic_id` | Target identifier. |
| `ephemeris` | `period_days`, `epoch`, `duration_days`, and `source` — `catalogue`, `bls` or `user`. |
| `prob_calibrated` | The calibrated probability. This is the number to act on. |
| `prob_mean`, `prob_std` | Mean and standard deviation over MC-Dropout passes — the model's own uncertainty. |
| `per_fold` | Each fold model's probability, so agreement across folds is visible. |
| `decision_threshold` | The operating point the verdict was rendered at. |

### 4.2 Diagnostic panels

The response carries the series a human vetter reads, so the console can render
evidence rather than a number alone.

| Field | Description |
|---|---|
| `global_view` | Phase-folded flux over the full orbit. |
| `local_view` | Phase-folded flux around the transit. |
| `odd_view`, `even_view` | Odd- and even-numbered transits folded separately. A depth difference indicates an eclipsing binary at twice the period. |
| `centroid_track` | Flux-weighted centroid motion through transit. Motion indicates the signal originates off-target. |
| `periodogram` | Frequency-domain power, for whether the period is real beyond the transit itself. |

### 4.3 Cautions

Each caution block reports its statistic, its threshold, and a boolean. They are
advisory and do not alter the probability.

| Block | What it tests |
|---|---|
| `centroid` | Centroid shift in transit — background eclipsing binary. |
| `odd_even` | Odd/even depth and timing difference — binary at twice the period. |
| `secondary` | Occultation at phase 0.5, with an albedo comparison — self-luminous companion. |
| `duration_check` | Observed duration against the circular-orbit expectation — implausible geometry. |
| `false_alarms` | SWEET test, transit asymmetry, depth mean-median ratio, gap fraction. BLS-found ephemerides only. |

A block is `null` when the raw light curve lacks the columns it needs. Absence
is reported rather than imputed, because a caution that silently defaults to
"clear" is worse than no caution at all.

## 5. Known Limitations

Stated here rather than in a footnote, because they bound what the outputs mean.

**The pipeline vets known ephemerides; it does not search.** It answers "is this
signal a planet", not "is there a planet here".

**The labels inherit the archive's biases.** Targets enter the training set only
once a human or pipeline has already dispositioned them, so the training
distribution is dispositioned candidates, not a random sample of observed stars.
Bright-star and short-period selection effects are built in.

**The model partly scores the star rather than the transit.** A zero-depth
control arm — synthetic transit-free light curves at real host positions — is
passed by a measurable fraction of hosts, and more often by planet hosts than
false-positive hosts. Score correlates with how long a target was observed more
strongly than with how much transit evidence was collected. This is measured,
tracked and partly reduced; it is not solved. Current figures are in
[data_provenance.md](data_provenance.md).

**Pixel-level information is largely absent.** Distinguishing an on-target
transit from a blended background eclipse ultimately requires the pixels. The
difference-image branch is outstanding work in [roadmap.md](roadmap.md).

**It has not been evaluated prospectively at scale.** The only honest test of a
vetting system is scoring candidates before their dispositions are published.
That clock started on 2026-07-25.

## 6. References

Bibliography in [references.bib](references.bib).
