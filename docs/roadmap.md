# Roadmap — the ExoMiner-inspired rebuild

Adopted 2026-07-26 after reviewing [NASA's ExoMiner](https://github.com/nasa/ExoMiner)
(ExoMiner++, TESS paper: [AJ 170, 5](https://iopscience.iop.org/article/10.3847/1538-3881/ae03a4)).
The UI redesign stays the locked final step.

We reimplement and credit; we do not vendor their code (NASA NOSA licence).

## Why ExoMiner

Its branches target pathologies we have *measured*, not guessed:

| our measurement | what it means | ExoMiner's answer |
|---|---|---|
| corr(prob, transit count) **−0.048** | the score does not track how many transits were actually caught | unfolded per-transit branch + observed/expected transit-count scalars |
| **26.4%** of hosts pass threshold with no injection at all (46.7% planet hosts vs 12.3% FP hosts) | the model partly scores the host, not the transit | per-diagnostic branches with branch-scoped scalars, so transit evidence is explicit |
| 13-dim vetting-aux retrain: ΔAUC **−1.3e-5** | scalar summaries of diagnostics add nothing over the views | feed the odd/even, secondary and centroid **views** |
| TESS **0.906** vs Kepler **0.989** AUC | the headroom is on the mission we serve | momentum dump, transit-masked periodogram, per-sector difference-image quality |

**The baseline correlation is no longer on this list.** It was, at +0.211, read as
the model scoring observation time rather than transit evidence. Measured
properly it is label structure, not a model pathology — see *Observation
baseline* under stage 3. The transit-count row above is also restated: the
original **−0.003** was measured against `expected_transit_count`, the transits
the ephemeris *predicts*; against the transits actually *captured* it is
**−0.048**. The conclusion survives the correction, the number does not.

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

**Stage 1 — ExoMiner-grade inputs.** *(done 2026-08-05)*
A 5,423-example view set with eleven branches — 301/31 flux with variance and
presence channels, odd/even, weak-secondary, centroid, flux-trend, unfolded
[20,31] with transit counts, cadence-gap, and the periodogram pair — plus a
3.6 GB DV archive, its scalars table, Gaia RUWE, FFI recovery for the 744
`no_fits` candidates, and a seventh validation gate. `ca906040` served
untouched throughout.

The three things flagged as likely to bite, and what actually happened:

- **Shard size.** Feared 20-50x. Actual **2.6x** (122 MB against 47 MB), so the
  `tf.data.cache()` decision needed no revisiting.
- **Per-branch presence masking.** Necessary exactly as predicted: `dv_usable`
  is 87.4% on TESS and **0% on Kepler and K2**. Every branch carries a presence
  channel and the model gates on it.
- **The DV download.** Sized at 14-56 GB and many hours; actual **3.6 GB** in
  5.3 h. The 2-8 MB/file estimate was the DVR *PDF* and DVT *FITS*, not the
  ~0.34 MB XML. What mattered for runtime was batching the availability query
  (40 TICs per round trip), not scoping sectors.

Two things could not be built as specified. The **momentum-dump branch** reads
`QUALITY` bit 5, which lightkurve's default bitmask strips at download — the
flag is zero on every cadence in the cache — so it measures the hole a dump
leaves instead. And **difference-image stamps are 11-17 px, not a fixed
33x33**; that is Kepler's size, and stage 2(d) must re-grid.

Full detail, and the merge collision that silently dropped the transit counts
past all seven gates, in HANDOVER.md (2026-08-05).

**Stage 2 — the model, incrementally, each sub-step gated.**
(a) per-diagnostic branches + scoped scalars + variance channels + joint local
conv; (b) unfolded-flux branch; (c) trend + periodogram branches;
(d) difference-image branch with quality attention. Optuna re-tune on the
winner. Every sub-step passes the promotion gate on CV AUC/Brier/ECE, the TESS
slice, and injection-recovery completeness.

**Stage 2(b)'s success criterion, re-specified 2026-08-05.** It read
*"corr(prob, n_transits) must leave zero and the 26.4% control-arm host-pass
rate must fall"*, with a companion requirement that the baseline correlation
fall from +0.211. That criterion is now split, because its two halves are not
the same kind of measurement:

- **The control-arm host-pass rate is the criterion.** It is measured on real
  hosts with *no injection*, so a pass means the model scored the star rather
  than a transit. No label structure enters it and nothing about the catalogue
  can explain it away. **26.4% must fall.**
- **The baseline correlation is retired as a gate** and kept only as a reported
  diagnostic. Driving it to zero would move the model away from its own labels
  — see stage 3.
- **The transit-count correlation is reported, not gated.** Its zero point is
  **−0.048** against transits captured, not the −0.003 that was measured against
  transits predicted; and the labels themselves sit at −0.073, so there is no
  defensible target value to demand.

The clean test of the unfolded branch is **injection-recovery on matched hosts
with observation baseline held constant**, which removes the label confound
entirely. Build that harness when 2(b) is run.

**Stage 3 — labels and negatives.** EB-catalogue and brown-dwarf negatives,
the ephemeris-match test, and scrambled/inverted synthetic negatives built with
our existing injection machinery. Plus the observation-selection problem below,
which arrived here from stage 2.

### Observation baseline — a real problem architecture cannot fix

Measured 2026-08-05, baseline as a span in **days**:

| population | corr(score, baseline) |
|---|---:|
| incumbent, 3,908 scored candidates | **+0.208** (+0.187 controlling period) |
| incumbent, labelled CV set | +0.238 |
| stage 2(a) branches, labelled CV set | +0.239 |
| **the ground-truth label itself** | **+0.278**, and **+0.387** on TESS alone |

Every model sits *below* the labels. The correlation survives inside every TESS
period band and is not a period artefact. TESS confirmed planets have a median
baseline of **1,495 d against 430 d** for false positives.

The mechanism is confirmation bias in the catalogue: a target observed across
many sectors accumulates the follow-up that promotes it to confirmed, while a
briefly-observed one stays a candidate or is retired. The model learned it
because in the training labels it is true.

**This is not "the correlation turned out to be fine".** It is a genuine defect
with the wrong owner. For the deployment use — ranking candidates for follow-up
— baseline dependence actively defeats the purpose, because it promotes targets
that already received attention over under-observed ones that may deserve it.
What changed is only *what can fix it*: no architecture can, because the signal
is in the labels. The levers are **propensity-score weighting on observation
baseline**, **baseline-stratified negative sampling**, and **synthetic negatives**
that break the correlation by construction. All three are label-distribution
interventions, and all three belong here.

**Stage 4 — serving parity and explainability.** `TargetScorer` computes every
branch live; `/score` returns per-branch contributions via branch-occlusion.
ExoMiner's explainability story, made interactive — which their batch pipeline
cannot do.

**Stage 5 — the UI redesign.** Unchanged and last. Mission Control aesthetic,
manus north star. It will have per-branch vetting evidence to display.

## Considered and deferred

Decisions recorded with their reasoning, so they are not re-litigated from
scratch each time they come up.

### Transit search on raw light curves

**Status: possible today at single-target scale; deferred as a programme.**

The distinction is between *detection* (finding a periodic dip in photometry
where no ephemeris is known) and *vetting* (deciding whether a known signal is
a planet). This project does vetting. But the detection machinery is already
present and works: `search/bls.py` provides `period_grid`,
`bls_period_search` and `bls_periodogram`, and `/score/{tic_id}?force_bls=true`
ignores the catalogue ephemeris and searches blind, so detection →
classification already runs end to end on one target.

What is missing is survey scale, which is a different engineering problem:

- **iterative multi-planet search** — mask the strongest signal, re-search the
  residual, repeat;
- **false-alarm control** — a blind search over ~5,000 trial periods will find
  peaks in pure noise, so a bootstrap or analytic FA probability is required
  (this is the still-open "statistical bootstrap FA", DV §3.5, in the review
  gaps);
- **compute** — ~200,000 two-minute targets per TESS sector and millions in the
  FFIs, against a BLS capped at 5,000 trial periods over ~20,000 cadences;
- **detrending at scale** that does not absorb the signal being searched for.

**Why it is deferred.** SPOC already runs a comprehensive detection pipeline
over all 2-minute data — that is where TOIs come from — so re-detecting what it
has already detected is unlikely to add anything. More importantly, the
measured weaknesses are all on the vetting side (the 26.4% control-arm host
pass rate; probability tracking transits captured at −0.048), and stages 1–3
have a falsifiable test for fixing them. Switching
to a harder, more crowded problem mid-solve would abandon a well-posed one.

**The niche worth remembering.** BLS assumes a repeating box, so it is weakest
exactly where this model already behaves oddly: long-period and single-transit
events. 66 of the 3,919 scored candidates have a baseline covering fewer than
two periods, and 6 of the top 20 have periods over 400 days. Single- and
duo-transit detection is an area where BLS structurally underperforms and where
the community is still finding planets. If search ever enters this project,
that is the door — not a general re-run of SPOC.

### A large language model in the pipeline

**Status: rejected for the core pipeline.**

Periodically tempting as on-device models improve (e.g. 1-bit/ternary builds
small enough to run locally). It fails the governing rule — there is no
baseline here that an LLM beats, because no task in the pipeline is
language-shaped. The core job is binary classification of a 2,001-bin global
view, a 201-bin local view and 13 scalars; a dual-view CNN does that at 0.958
AUC in milliseconds. Transit shape is a continuous-signal problem, not a
token-sequence one.

The deployment maths is independently disqualifying: the API runs on a 2 GB Fly
machine (peak 548 MB) at roughly $1–4/month scale-to-zero, and the smallest
useful local model is several GB of weights before its KV cache.

Adjacent uses that are *not* the pipeline: literature cross-referencing for a
shortlisted target is genuinely useful and belongs in a separate tool.
Natural-language verdict summaries are already produced deterministically from
the structured diagnostics. For explainability, stage 4's per-branch occlusion
contributions are quantitative and faithful to what the model computed;
narrating them with an LLM would add a layer that can be wrong about our own
model, which is the opposite of the goal.
