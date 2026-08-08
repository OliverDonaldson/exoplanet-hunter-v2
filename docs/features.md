# Model inputs

What the promoted model actually consumes, and where each value comes from.
This is the current state — the ExoMiner-inspired rebuild (roadmap stages 1–2)
replaces the single aux vector with per-diagnostic branches, and this table is
the baseline it will be measured against.

Written in the style of [ExoMiner's feature table](https://github.com/nasa/ExoMiner/blob/main/docs/exominer-features.md).

## Views

Phase-folded at the ephemeris (catalogue where published, BLS otherwise), then
median-binned. The transit is masked out of the Savitzky-Golay fit so the
detrending cannot absorb the dip.

| feature | dim | dtype | source |
|---|---|---|---|
| `global_views` | [2001] | float32 | full phase [-0.5, 0.5] |
| `local_views` | [201] | float32 | ±3 transit durations around phase 0 |

Config: `pipeline/conf/preprocess/default.yaml`.

**Known gap.** ExoMiner uses 301/31 bins and pairs every view with a variance
channel; ours are ~7× oversampled with no variance, so per-bin noise is higher
and the model cannot see bin dispersion. Stage 2 of the roadmap *(old stage 1)*
addresses both.

## Auxiliary vector

One 13-dim vector, imputed → log-transformed on the heavy-tailed columns →
standardised, fitted per fold and persisted in the calibration bundle
(`features/aux.py::build_aux_row` is the single implementation, shared by
training and serving).

| idx | feature | unit | source |
|---:|---|---|---|
| 0 | `teff` | K | TIC-8 / catalogue stellar params |
| 1 | `radius` | R☉ | TIC-8 / catalogue |
| 2 | `logg` | log₁₀(cm/s²) | TIC-8 / catalogue (≈half of K2 rows imputed) |
| 3 | `tmag` | mag | TIC-8 / catalogue |
| 4 | `depth` | fraction | catalogue ephemeris |
| 5 | `duration` | days | catalogue ephemeris |
| 6 | `log_period` | log days | catalogue ephemeris |
| 7 | `pink_snr` | σ | computed from the light curve (LEO-Vetter §2.1) |
| 8 | `centroid_snr` | σ | MOM_CENTR motion test (DV §3.6) |
| 9 | `oe_depth_sigma` | σ | odd/even depth difference |
| 10 | `oe_timing_sigma` | σ | odd/even midtime difference (Eq 13) |
| 11 | `secondary_sig` | σ | box-scan Model-Shift secondary |
| 12 | `q_ratio` | — | duration / circular-orbit duration |

Indices 7–12 are the vetting features added in the 13-dim build. Promoted runs
that predate them (including the live `ca906040`) serve a 9-dim layout;
`LEGACY_AUX_DIM` keeps them loadable and byte-identical.

**Measured caveat.** Adding 9–12 produced a statistically indistinguishable
model (ΔAUC −1.3e-5). Their information is largely already in the folded views,
and they were added as *scalar summaries* of diagnostics whose signal lives in
shapes. That result is the main argument for the branch-per-diagnostic
restructure: ExoMiner feeds the odd/even, secondary and centroid **views**, not
their summary statistics.

## Not currently used

Available upstream and adopted in the roadmap, but absent today:

| input | why it matters | stage |
|---|---|---|
| Difference images (33×33×N) | pixel-level source location; the strongest nearby-EB discriminant | 1–2 |
| DV diagnostics (bootstrap FAP, ghost core/halo, MES, χ², robust stat) | SPOC's own vetting statistics | 1–2 |
| Gaia RUWE | astrometric excess noise — unresolved binaries | 1 |
| Unfolded per-transit views | whether the signal recurs, vs one artefact | 1–2 |
| Momentum-dump view | TESS-specific systematic | 1–2 |
| Flux-trend view | stellar variability confusing the fit | 1–2 |
| Periodogram (normal + transit-masked) | is the period real beyond the transit itself | 1–2 |

## Labels

| disposition | label | source |
|---|---|---|
| CP, KP (TESS) / CONFIRMED (Kepler, K2) | 1 | NASA Exoplanet Archive |
| FP, FA (TESS) / FALSE POSITIVE, REFUTED | 0 | archive + DR25 certification |
| PC, CANDIDATE | −1 (held out) | not trained on |

Kepler negatives are restricted to DR25-certified false positives
(`koi_score < 0.5`). See `docs/data_provenance.md`.
