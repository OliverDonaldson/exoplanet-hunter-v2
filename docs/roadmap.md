# Roadmap — the ExoMiner-inspired rebuild

Adopted 2026-07-26 after reviewing [NASA's ExoMiner](https://github.com/nasa/ExoMiner)
(ExoMiner++, TESS paper: [AJ 170, 5](https://iopscience.iop.org/article/10.3847/1538-3881/ae03a4)).
The UI redesign stays the locked final step.

We reimplement and credit; we do not vendor their code (NASA NOSA licence).

## Why ExoMiner

Its branches target pathologies we have *measured*, not guessed:

| our measurement | what it means | ExoMiner's answer |
|---|---|---|
| corr(prob, baseline) **+0.211**, corr(prob, transit count) **−0.003** | the score tracks how long a target was observed, not how many transits were caught | unfolded per-transit branch + observed/expected transit-count scalars |
| **26.4%** of hosts pass threshold with no injection at all (46.7% planet hosts vs 12.3% FP hosts) | the model partly scores the host, not the transit | per-diagnostic branches with branch-scoped scalars, so transit evidence is explicit |
| 13-dim vetting-aux retrain: ΔAUC **−1.3e-5** | scalar summaries of diagnostics add nothing over the views | feed the odd/even, secondary and centroid **views** |
| TESS **0.906** vs Kepler **0.989** AUC | the headroom is on the mission we serve | momentum dump, transit-masked periodogram, per-sector difference-image quality |

## What we take, and what we do not

**Adopt.** Per-diagnostic conv branches with scoped scalars; paired variance
channels; unfolded-transit branch; secondary/centroid/trend/periodogram views;
301/31 median-binned views (ours are ~7× oversampled); train-shard-only
normalisation statistics; AUC-PR early stopping; DV XML ingest (difference
images, DV scalars, Gaia RUWE); their documentation structure.

**Do not.** Their podman batch pipeline — we serve live and interactively.
Focal loss at α=0.96 — tuned for a ~2% TCE base rate; ours is ~50/50 and our
own Optuna campaign rejected focal. Their static train/test split — our
injection-recovery, control arm and since-confirmed holdout are a stronger
evaluation than they publish, and remain the gate.

## Stages

**Stage 0 — housekeeping, landmines, vendoring.** *(done)*
71 GB of stitched-and-forgotten staging deleted and auto-cleanup added;
`preprocess_only.py` and `score_target.py` deleted (both could silently write
9-dim/no-K2 data or break on a non-9-dim model); the four remaining audit items
fixed (gate cwd, `dvc` resolution, MLflow run naming, the CI gate jobs
`ci.yml` had promised); patched TRICERATOPS vendored.

**Stage 1 — ExoMiner-grade inputs.** *(data only; one rebuild run)*
Emit 301/31 views with variance channels, odd/even, weak-secondary, centroid,
flux-trend, unfolded [20,31] + transit counts, momentum dump, and the
periodogram pair. Ingest TESS DV XML for difference images, DV diagnostics and
RUWE; per-branch presence masking (Kepler has DV, K2 does not). Add the
TESS-SPOC FFI fallback to recover part of the 744 `no_fits` candidates. Extend
the validation gates to the new schema.

**Stage 2 — the model, incrementally, each sub-step gated.**
(a) per-diagnostic branches + scoped scalars + variance channels + joint local
conv; (b) unfolded-flux branch — success is specific: corr(prob, n_transits)
must leave zero *and* the 26.4% control-arm host-pass rate must fall;
(c) trend + periodogram branches; (d) difference-image branch with quality
attention. Optuna re-tune on the winner. Every sub-step passes the promotion
gate on CV AUC/Brier/ECE, the TESS slice, and injection-recovery completeness.

**Stage 3 — labels and negatives.** EB-catalogue and brown-dwarf negatives,
the ephemeris-match test, and scrambled/inverted synthetic negatives built with
our existing injection machinery.

**Stage 4 — serving parity and explainability.** `TargetScorer` computes every
branch live; `/score` returns per-branch contributions via branch-occlusion.
ExoMiner's explainability story, made interactive — which their batch pipeline
cannot do.

**Stage 5 — the UI redesign.** Unchanged and last. Mission Control aesthetic,
manus north star. It will have per-branch vetting evidence to display.
