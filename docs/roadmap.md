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

**Stage 1 — ExoMiner-grade inputs.** *(done 2026-08-05)*
Built: 5,423-example view set (11 branches), 3.6 GB DV archive, DV scalars
table, Gaia RUWE, FFI recovery for the `no_fits` candidates, and a seventh
validation gate. Shards are 122 MB against the legacy 47 MB — 2.6x, so the
`.cache()` question needed no change. Details and the four corrections in
HANDOVER.md (2026-08-05).

Original plan:
Emit 301/31 views with variance channels, odd/even, weak-secondary, centroid,
flux-trend, unfolded [20,31] + transit counts, momentum dump, and the
periodogram pair. Ingest TESS DV XML for difference images, DV diagnostics and
RUWE. Add the TESS-SPOC FFI fallback. Extend the validation gates to the new
schema.

*No model change in this stage, deliberately.* Changing inputs and
architecture together makes any outcome uninterpretable — the 13-dim null was
only diagnosable because the aux vector was the sole variable. The data build
is also the expensive, network-bound half (hours of MAST), while training
reads shards from disk, so the ordering pays the slow step once and then
iterates cheaply. And `cnn_dualview.py` declares exactly three Input layers
(`global_view`, `local_view`, `aux_features`) — it *cannot* consume an
unfolded [20,31] tensor or a 33x33xN image stack. Consuming them is Stage 2.

Nothing reaches the promotion gate here either: that gate compares trained
runs' CV summaries, and this stage produces data, not a model. Stage 1 is
gated by the *five Pandera gates* instead, which is why extending them is a
deliverable rather than a nicety — the new artefacts are exactly where silent
corruption would hide, and this project has already been bitten three times by
data that was wrong but plausible (the 500/500 catalogue clobber, the
9-dim-into-13-dim write, the ppm-vs-fraction depth). `ca906040` keeps serving
untouched throughout.

### Stage 1 — three things that will bite

**Shard size.** TFRecords are 47 MB today (`views.npz` 36 MB, 5,380 examples x
[2001 + 201 + 13]). Unfolded views and 33x33xN image stacks could plausibly
push that **20-50x**, which changes what fits in memory and how tf.data should
be configured (`.cache()` on a 2 GB artefact is not the same decision as on
47 MB). Measure after the first few hundred targets, not after the full build.

**Per-branch presence masking is not optional.** Kepler has DV products, K2 has
none, TESS FFI differs again. Without an explicit presence/quality mask per
branch, a missing branch poisons every row of that mission — the same class of
silent, plausible-looking corruption as the 9-dim-into-13-dim write.
ExoMiner's difference-image quality attention is the pattern to copy.

**~~The DV download is the biggest single fetch in the project's history.~~**
*(Done 2026-08-01 — and it was not. Actual: **3.5 GB**, 5.3 h, 7,199 targets.)*

The 2026-07-31 estimate of 14-56 GB was ~5x high: a DV XML is **~0.34 MB**, and
2-8 MB was the DVR *PDF* (18-21 MB) and DVT *FITS* (11-22 MB). Availability came
in at **80.0%**, so ~1,440 targets have no DV products and need the presence
mask. Sector scope turned out to be on disk already — in the `sectors` column of
`data/catalogue/candidates.parquet` (7,195 of 7,199), not the download manifest,
which records `n_sectors` as a count only. What actually mattered for runtime was
**batching**: `query_criteria` takes a list of `target_name`s, and 40 per round
trip is 0.29 s/target against 1.8 s. The rule that held: **do not run it while
another MAST job is going**, and write it resumable and manifest-tracked from
the start — 82 transient failures on the first pass were swept by a re-run.

**The FFI fallback has upside beyond the model.** `TESS-SPOC` HLSP could
recover a real fraction of the **744 `no_fits`** candidates — targets
currently invisible to the entire pipeline, not merely poorly scored. That is
the one Stage 1 item whose value does not depend on Stage 2 succeeding.

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
pass rate; probability tracking observation baseline at +0.21 but transit count
at −0.003), and stages 1–2 have a falsifiable test for fixing them. Switching
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
