# Model Specifications

What the models consume, what shape they are, and where every input comes from.
Two model families are in the repository: the **dual-view** model, which is
currently served, and the **branch** model, which is the ExoMiner-inspired
rebuild. Both are described here because results are reported against both.

## 1. Inputs

### 1.1 Views

Light curves are phase-folded at the ephemeris — from the catalogue where
published, from a BLS search otherwise — then median-binned. The transit is
masked out of the Savitzky-Golay fit, so detrending cannot absorb the dip it
exists to preserve.

The dual-view model takes two views:

| View | Shape | Coverage |
|---|---|---|
| `global_view` | [2001] | full phase, [-0.5, 0.5] |
| `local_view` | [201] | ±3 transit durations around phase 0 |

The branch model takes eleven, each with its own convolutional branch. The third
channel is a per-bin variance channel; `gap_view` and the periodograms carry two.

| View | Shape | What it is for |
|---|---|---|
| `global_view` | [2001, 3] | full orbit |
| `local_view` | [201, 3] | the transit itself |
| `odd_view` | [201, 3] | odd-numbered transits only |
| `even_view` | [201, 3] | even-numbered transits only |
| `secondary_view` | [201, 3] | centred on the weak-secondary phase |
| `trend_view` | [2001, 3] | the detrending residual — stellar variability |
| `centroid_view` | [201, 3] | flux-weighted centroid motion through transit |
| `unfolded_view` | [20, 201, 3] | up to 20 individual transits, unstacked |
| `gap_view` | [2001, 2] | observation coverage and cadence gaps |
| `periodogram_view` | [256, 2] | frequency-domain power |
| `periodogram_masked_view` | [256, 2] | the same with the transit masked out |

`unfolded_view` is the structural difference from the dual-view design: it lets
the model see whether a signal *recurs*, rather than only the stacked average of
everything that recurred.

Configuration: `pipeline/conf/preprocess/default.yaml`.

### 1.2 Auxiliary vector

One 13-dimensional vector, imputed, log-transformed on the heavy-tailed columns,
then standardised. Fitted per fold and persisted in the calibration bundle.
`features/aux.py::build_aux_row` is the single implementation, shared by training
and serving, so the two cannot drift.

| idx | Feature | Unit | Source |
|---:|---|---|---|
| 0 | `teff` | K | TIC-8 / catalogue stellar parameters |
| 1 | `radius` | R☉ | TIC-8 / catalogue |
| 2 | `logg` | log₁₀(cm/s²) | TIC-8 / catalogue (~half of K2 rows imputed) |
| 3 | `tmag` | mag | TIC-8 / catalogue |
| 4 | `depth` | fraction | catalogue ephemeris |
| 5 | `duration` | days | catalogue ephemeris |
| 6 | `log_period` | log days | catalogue ephemeris |
| 7 | `pink_snr` | σ | computed from the light curve |
| 8 | `centroid_snr` | σ | MOM_CENTR motion test |
| 9 | `oe_depth_sigma` | σ | odd/even depth difference |
| 10 | `oe_timing_sigma` | σ | odd/even midtime difference |
| 11 | `secondary_sig` | σ | box-scan model-shift secondary |
| 12 | `q_ratio` | — | duration / circular-orbit duration |

Indices 7–12 are the vetting features added in the 13-dim build. Runs predating
them, including the served model, use a 9-dim layout; `LEGACY_AUX_DIM` keeps
those loadable and byte-identical.

**Measured caveat.** Adding indices 9–12 produced a statistically
indistinguishable model (ΔAUC −1.3×10⁻⁵). Their information is largely already
present in the folded views, and they were added as *scalar summaries* of
diagnostics whose signal lives in shapes. That null result is the main argument
for the branch-per-diagnostic restructure: the odd/even, secondary and centroid
**views** are fed directly, not their summary statistics.

### 1.3 Labels

| Disposition | Label | Source |
|---|---|---|
| CP, KP (TESS) / CONFIRMED (Kepler, K2) | 1 | NASA Exoplanet Archive |
| FP, FA (TESS) / FALSE POSITIVE, REFUTED | 0 | archive + DR25 Robovetter score |
| PC, CANDIDATE | −1, held out | not trained on |

Kepler negatives are restricted to DR25 FALSE POSITIVE KOIs with a Robovetter
disposition score below 0.5 — a majority false-positive vote under perturbation.
That is a reconstruction of, not the same thing as, the Kepler Certified False
Positive table, which is no longer served by the archive. See [data_provenance.md](data_provenance.md).

## 2. Architecture

### 2.1 Dual-view model

Two 1-D convolutional towers, one per view, concatenated with the auxiliary
vector and passed through a dense head. Trained with 5-fold cross-validation,
MC-Dropout for uncertainty, and Platt scaling for calibration.

Platt, not temperature scaling: temperature has no bias term and cannot correct
a distribution shift, which cost a 0.136 ECE regression before it was fixed.

### 2.2 Branch model

One convolutional branch per view, each reduced to a `branch_units` embedding,
concatenated with scoped scalars and passed through a shared head.

| Parameter | Value |
|---|---|
| `conv_blocks` | 2 |
| `init_filters` | 16 |
| `kernel_size` | 5 |
| `pool_size` | 4 |
| `branch_units` | 32 |
| `head_units` | [256, 128] |
| `dropout` | 0.3 |

`drop_branches` ablates named branches for attribution runs without retraining
the rest of the design.

### 2.3 Ensembling

Both trainers accept `n_models_per_fold`. Members are trained independently,
their scores averaged, and the calibrator fitted **once over the average**. Each
member's uncalibrated score is written as `member_score_N`, so the spread the
run averaged over remains readable after the fact. That spread is the noise
floor every margin in this project is read against.

At one member per fold the path is bit-identical to every run made before
multi-member training existed, including the served model's.

## 3. Not currently used

Available upstream, absent from the served model today.

| Input | Status |
|---|---|
| Difference images | built and tested; carries no measured signal — see below |
| Gaia RUWE | never built; astrometric excess noise, an unresolved-binary discriminant |

**Difference images are no longer an open gap.** They were the largest one until
2026-09-05. The branch was built (stage 9), the stamps re-gridded from the 11–17
px they actually are rather than the fixed 33x33 the design assumed, and the one
remaining explanation for stage 9's null — that the stamps were fed without the
star's own pixel position, so the network had no reference frame for a centroid
shift — was built and tested as Phase 1. The paired contrast moved -0.0254 on
TESS recall @1% FPR on `dv_usable` rows against a floor of 0.0979, which is 0.26x
its floor, so the explanation is **falsified** and no third stamp variant is
commissioned. See [experiments/phase-1-result.md](experiments/phase-1-result.md)
and the closing entry in [decisions.md](decisions.md).
