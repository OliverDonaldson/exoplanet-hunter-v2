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
median-binned views; train-shard-only normalisation statistics; AUC-PR early
stopping; DV XML ingest (difference images, DV scalars, Gaia RUWE); their
documentation structure.

**Their 301/31 bin counts: suspected, tested, exonerated.** At 301/31 run 1's
Kepler deficit rose with transits caught, reaching +0.1446 where a narrow
transit is folded from many of them — a clean-looking resolution signature.
Restoring 2001/201 in run 2 **made everything worse** (Kepler gap +0.0348 →
+0.0707). The correlation was real and the causal reading of it was wrong.
301/31 is not what is holding the branch model back, and the bin counts stay
where ExoMiner put them unless something new implicates them.

**Do not.** Their podman batch pipeline — we serve live and interactively.
Focal loss at α=0.96 — tuned for a ~2% TCE base rate; ours is ~50/50 and our
own Optuna campaign rejected focal. Their static train/test split — our
injection-recovery, control arm and since-confirmed holdout are a stronger
evaluation than they publish, and remain the gate.

## Where the project stands

One table, kept current. Detail for each row is in the stage sections below and
in HANDOVER.md.

| stage | status | what closed it, or what is left |
|---|---|---|
| **0** housekeeping, landmines | **done** | 71 GB staging reclaimed; two scripts that could silently write bad data deleted; four audit items fixed; TRICERATOPS vendored |
| **1** ExoMiner-grade inputs | **done** 2026-08-05 | 5,423 examples × 11 branches, 3.6 GB DV archive, DV scalars, Gaia RUWE, FFI recovery, seventh gate |
| **2(a)** per-diagnostic branches | run 1 **REJECTED**; run 2 **REJECTED** | run 1 lost +0.0222 all-mission. Run 2 restored 2001/201 and lost **+0.0516** — resolution hypothesis falsified; the capacity run is now mandatory |
| **2(b)** unfolded-flux branch | **blocked on 2(a)** | criterion re-specified 2026-08-05: control-arm host-pass rate, not the baseline correlation |
| **2(c)** trend + periodogram | not started | |
| **2(d)** difference-image branch | not started | needs the 11–17 px stamps re-gridded to a fixed size |
| **3** labels and negatives | not started | now also owns the observation-selection problem, which arrived from stage 2 |
| **4** serving parity + explainability | not started | branch-occlusion contributions through `/score` |
| **5** UI redesign | locked last | |

**Serving is unchanged throughout: `ca906040` (9-dim, 2001/201) on Fly.** Nothing
in stage 1 or 2 has been promoted, and the registry has not been touched since
2026-07-19.

### Uncovered, fixed, and improved since stage 1 closed

Findings, not features — each one is something that was silently wrong, or
silently unmeasured, and is now neither.

| what was wrong | how it surfaced | where the fix lives |
|---|---|---|
| **K2 — 9.7% of training — had never been benchmarked**, and the incumbent comparison dropped all 527 rows silently | per-mission coverage of the "identical rows" join | `scripts/evaluate.py score`, `eval/comparison.py` |
| Comparing two prediction sets reported no coverage, so a whole mission could vanish from a decision | the above | `mission_coverage`, `compare_prediction_sets(strict=)` |
| The run-1 reading — per-mission gaps, and the gap by transits caught — existed only as a session, not as anything re-runnable | needing the same three measurements for run 2 | `scripts/evaluate.py compare`, validated to reproduce every run-1 number |
| **The branch trainer saved no checkpoint at all** — run 1 scored weights that existed only in memory, leaving nothing to promote, rescore or serve | audit sweep against `train.py`'s "score what ships" rule | `train_branches.py` writes and reloads a per-fold checkpoint + bundle |
| The branch model could not be reloaded without `safe_mode=False`, because gating used a `Lambda` over a Python lambda | the checkpoint fix exposed it immediately | `PresenceFlag` / `PickColumns`, registered serializable layers |
| The gate ranked on AUC only, so a model could match on AUC and lose the shortlist | run 1: TESS AUC +0.002, recall @1% FPR −0.069 | recall @1% FPR is a gate criterion in `validation/promotion.py` |
| **A NaN metric promoted.** Every guard is an inequality and NaN loses all of them, so a degenerate run would report `ROC-AUC nan vs incumbent 0.9581` and **PROMOTE** | pre-flighting the new CV path on a single-class subset | finite-check runs before every comparison in `evaluate_promotion` |
| The gate decided on an aggregate whose weights are a sampling artefact | Kepler is drawn at exactly 1,250/1,250 | gate reads TESS; Kepler/K2 alarmed; aggregate reported only |
| Bin counts were declared twice — in the builder and in the shard schema | restoring 2001/201 would have needed two edits that could disagree | `VIEW_SHAPES` derives from the builder's constants |
| The per-target view cache was not keyed by resolution, so a rebuild would read back old shapes | same change | `_cache_path` keys on `g{GLOBAL_BINS}l{LOCAL_BINS}` |
| Run 1 trained **without augmentation** against an incumbent that had it | audit of what made the comparison unlike-for-like | `datasets/viewset_augment.py`, on by default |
| **The training noise floor had never been measured**, and the `±` in every `cv_summary.json` was being read as the run's uncertainty when it is the spread across folds within one run | run 2's fold 0 not reproducing in a diagnostic | quantified: single-fold sd **0.0106** over 5 repeats; recorded under stage 2(a) |

| **The incumbent had never been scored on the current view set**, so every stage-2 comparison ran against a 2026-07-19 baseline whose 4,818 rows predate K2, the DV scalars and the merge-collision fix | auditing which rows each `cv_summary.json` mean was computed over | `eval/scoring.py`, re-baselined set at `results/incumbent_rebaselined.parquet` |
| **The TESS gate never actually engaged.** `_gate_slice` returns `None` for a summary without a `per_mission` block, and the live incumbent has none — so every stage-2 decision silently fell back to comparing pooled means over different populations | regenerating the gate's decision from the real artefacts | `_population_mismatch` refuses the unmatched pooled comparison in `validation/promotion.py` |

| **Odd and even transits went through separate conv towers**, so the depth difference the head reads was partly a difference between two independently-learned sets of kernels — in the branch this model exists to make | reviewing ExoMiner's `build_joint_local_conv_branches` | one `TimeDistributed` tower over the flux family; fusion takes `odd - even` |
| A fold's score was a single seed draw, and the `±` in every summary mixed seed variance with fold difficulty with no way to separate them | the 0.0106 noise floor having no home in the artefact | `CVConfig.n_models_per_fold`; `summary.variance` reports `seed_sd` and `fold_sd` apart |

Tests: **304 → 428** (396 pipeline + 32 api). ruff and mypy clean on the
pre-commit config. Seven data gates pass.

### The shared flux tower — 2026-08-07

`local_view`, `odd_view`, `even_view` and `secondary_view` are the same
measurement at 201 bins, and they now pass through **one** conv tower
(`SHARED_LOCAL_VIEWS`). Fusion takes `odd - even` rather than the two
embeddings, gated on both halves being measured, and `odd_even_statistic` is
scoped to that contrast rather than to either half alone.

An eclipsing binary is the alternating-depth case; a subtraction only means
anything under tied weights. The model drops **233,617 → 215,281 parameters**,
which also removes the confound that muddied run 2: it is now *below* the
incumbent's 227,641 rather than above it, so a Kepler gain can no longer be
read as bought capacity. `centroid_view` is the same shape but carries a pixel
shift in units of its own scatter, so it is not comparable and keeps its tower.

### Audit of the recorded numbers — 2026-08-07

Every numeric claim in this file was regenerated from the artefacts on disk.
**All of run 1's numbers reproduce exactly**: the headline, both quartile
ladders and their trend slopes, all ten absolute bands with their row counts,
all five span-by-count cells, the K2 benchmark and the noise floor. The metric
implementations were checked against sklearn and against a brute-force sweep of
every achievable threshold — zero difference on all eight.

Two corrections came out of it, both recorded inline above: a Kepler Q2 quartile
gap misprinted as +0.0336 (actual +0.0335), and a false claim about the TESS
low-count cells. Nothing that changes a decision.

**Run 2 against run 1, paired fold by fold** (`validation.promotion.paired_folds`,
adopted from ExoMiner's `compute_confidence_interval.py`). Pairing on fold index
removes fold difficulty — the larger variance source, and the one identical
between two runs on the same split:

```
mean -0.0313, won 0/5, d=-1.92     deltas -0.042 -0.049 -0.018 -0.037 -0.010
```

**Run 2 lost every fold.** That is a materially stronger falsification than the
difference of two means, which is all the gate compared before. Quoted with its
caveat: three FFI rows arrived between the builds, so fold *k* is not quite the
same row set and the pairing is flagged inexact — three rows in 1,085 cannot
move a fold AUC by 0.01–0.05, but the guard says so rather than assuming.

Reported, not gating. At five folds the two-sided Wilcoxon floors at p=0.0625
and can never reach 0.05, so a gate keyed on it would reject every real
improvement too; `MIN_PAIRS_FOR_P_VALUE` suppresses the p-value entirely below
six pairs rather than inviting "not significant" to be read as "no effect".

**The re-baselined comparison**, incumbent scored on the current view set —
out-of-fold where it trained, zero-shot where it did not, never pooled across
the two. TESS is 100% out-of-fold on both sides, so the gating slice is exact:

| slice | n | incumbent | run 2 | gap |
|---|---:|---:|---:|---:|
| K2 *(first ever measurement)* | 527 | 0.9348 | 0.8741 | +0.0607 |
| Kepler | 2,481 | 0.9915 | 0.9230 | +0.0685 |
| **TESS** *(gates)* | 2,367 | 0.9100 | 0.8944 | **+0.0156** |
| all | 5,375 | 0.9523 | 0.9016 | +0.0507 |

Coverage rose from 4,605 rows across two missions to 5,375 across three. The
incumbent's own numbers barely moved (Kepler 0.9914 → 0.9915, TESS unchanged),
so the re-baseline **validates the earlier reading rather than overturning it** —
run 2 loses on every mission, including the one it had 527 training rows for and
the incumbent had none.

## Stages

**Stage 0 — housekeeping, landmines, vendoring.** *(done)*
71 GB of stitched-and-forgotten staging deleted and auto-cleanup added;
`preprocess_only.py` and `score_target.py` deleted (both could silently write
9-dim/no-K2 data or break on a non-9-dim model); the four remaining audit items
fixed (gate cwd, `dvc` resolution, MLflow run naming, the CI gate jobs
`ci.yml` had promised); patched TRICERATOPS vendored.

**Stage 1 — ExoMiner-grade inputs.** *(done 2026-08-05)*
A 5,423-example view set with eleven branches — global/local flux with variance
and presence channels, odd/even, weak-secondary, centroid, flux-trend, unfolded
per-transit with transit counts, cadence-gap, and the periodogram pair — plus a
3.6 GB DV archive, its scalars table, Gaia RUWE, FFI recovery for the 744
`no_fits` candidates, and a seventh validation gate. `ca906040` served
untouched throughout.

The three things flagged as likely to bite, and what actually happened:

- **Shard size.** Feared 20-50x. Actual **2.6x** at 301/31 (122 MB against
  47 MB), so the `tf.data.cache()` decision needed no revisiting *at that
  resolution*. Restoring 2001/201 moves it — see the run-2 sizing below.
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

### Stage 2(a) run 1 — REJECTED (2026-08-05)

**The stop condition fired.** Ollie's third pre-committed case, recorded before
the run: the branch model's all-mission gap is **+0.0222 in the incumbent's
favour on the 4,605 rows both models score**, against a fold standard deviation
of ~0.006. **It beats the incumbent nowhere.** `ca906040` stays served; nothing
was promoted and the registry is untouched.

**The mechanism, measured rather than assumed.** The loss is not spread evenly:

| slice | n | incumbent | branches | gap |
|---|---:|---:|---:|---:|
| TESS | 2,367 | 0.9100 | 0.9079 | **+0.0021** (level) |
| Kepler | 2,238 | 0.9914 | 0.9566 | **+0.0348** |

Almost the whole deficit is Kepler, and the Kepler gap is **monotonic in
transits caught**, by quartile of that count:

| quartile | 1 (fewest) | 2 | 3 | 4 (most) |
|---|---:|---:|---:|---:|
| Kepler gap | +0.0245 | +0.0335 | +0.0416 | **+0.0944** |

TESS is flat on the same split (+0.0002 per quartile step against Kepler's
+0.0218), and against observation *baseline* rather than transit count the
Kepler trend disappears.

**But "TESS is flat" is the wrong lesson, and the quartile split is what makes
it look right.** Quartiles are cut per mission, so TESS's top quartile is a
median of 89 transits where Kepler's is 1,035 — TESS looks immune largely
because it never reaches the regime. Cutting both missions on the *same
absolute* bands (measured 2026-08-06):

| transits caught | Kepler gap | TESS gap |
|---|---:|---:|
| 0–10 | +0.0269 (n=101) | +0.0109 (n=557) |
| 10–30 | +0.0253 (n=206) | −0.0091 (n=873) |
| 30–100 | +0.0243 (n=532) | +0.0034 (n=680) |
| 100–300 | +0.0390 (n=629) | +0.0024 (n=195) |
| **300+** | **+0.0690** (n=770) | **+0.0866** (n=62) |

**The deficit is a function of transits caught, not of mission.** Where TESS
does reach 300+ transits it shows the largest gap in the table — larger than
Kepler's, though on only 62 rows, so treat the point estimate loosely.

**The sharpest form: it is an interaction, not one variable.** Sorting by how
many of the 301 bins the transit itself spans (`duration / period × 301`), the
gap grows *with* span — the opposite of a "transit too narrow to resolve"
story. But span and count correlate (ρ 0.44), and crossing them separates the
two (Kepler):

| transit span | transits caught | n | incumbent | branches | gap |
|---|---|---:|---:|---:|---:|
| <4 bins | <100 | 690 | 0.9835 | 0.9631 | +0.0204 |
| **<4 bins** | **100+** | **149** | **0.9852** | **0.8406** | **+0.1446** |
| 4–8 bins | <100 | 117 | 0.9952 | 0.9639 | +0.0313 |
| 4–8 bins | 100+ | 413 | 0.9942 | 0.9399 | +0.0543 |
| 8+ bins | 100+ | 837 | 0.9915 | 0.9317 | +0.0597 |

Within *every* span band the gap still tracks transit count, and the worst cell
by a factor of three is **narrow in phase and caught many times**: 149 Kepler
targets with a median transit spanning **3.1 of 301 bins** and a median of
**134 transits caught**, where the incumbent scores 0.9852 and the 301-bin model
0.8406. At 2001 bins those same transits span ~21 bins. TESS shows the same
shape where it reaches the regime: its 8+ span / 100+ count cell (n=207) is
+0.0425, against −0.0086 and −0.0059 in the two wider-span low-count cells.

*(Corrected 2026-08-07 by regeneration from the artefacts. This previously read
"every low-count TESS cell sits at or below zero", which is false: the narrow-span
low-count cell is **+0.0243 on n=417**, the largest of the three. It weakens the
interaction reading — on TESS a narrow span alone carries a gap without a high
transit count — though run 2 falsified the resolution hypothesis regardless.)*

That is a coherent resolution mechanism rather than a hand-wave: many folded
transits make the per-bin median precise enough for fine structure to exist in
the data, and a coarse grid then smears it away — worst where the feature is
narrowest in phase.

**Pre-registered consequence.** If resolution is the cause, the **+0.1446 cell
improves most** in run 2, and the TESS 300+ band improves too. If the gap is
flat across these cells after the change, the effect is something other than
resolution and the reading below should be treated as falsified regardless of
what the headline Kepler number does.

**Both facts stand, and the second does not undo the first.** The mechanism
refines *why* the model lost; it does not un-fire the stop condition. The
resolution restoration below is a **new registered experiment with its own
pre-registered reading**, not a rescue of this run. Run 1 is rejected and stays
rejected whatever the resolution run returns.

Also not like-for-like, and recorded so the comparison is not overread: the
incumbent is Optuna-tuned and trained with augmentation on 2001/201 views; run 1
was a first pass at 193k params, two conv blocks, **no augmentation and no
tuning**. Augmentation now exists for the view set (`viewset_augment.py`) and is
declared in `conf/model/cnn_branches.yaml` at the same magnitudes the incumbent
trained with. Tuning stays out of run 2 deliberately — it would confound the
resolution test, and it is the Optuna step at the end of stage 2.

### Pre-commitments recorded before the next result exists

Written down first so they cannot be adjusted to fit an outcome.

**(a) The capacity re-run is CANCELLED, not deferred.** Recorded prediction:
`init_filters=22` — 226,711 parameters, 0.4% from the incumbent's 227,641 —
would move the Kepler gap by **less than 0.005**. Capacity is not the binding
constraint at a 0.0348 gap.

**MANDATORY TRIGGER, recorded with it:** if the resolution fix fails to close
the Kepler gap, the capacity run becomes **obligatory**, not optional.
Cancelling a pre-registered run after an unfavourable result is a red flag by
construction, and this guard is the whole reason the claim stays falsifiable.
The guard is the commitment; the cancellation is only its consequence.

Note also that the resolution change moves the parameter count on its own, so it
**partially subsumes the capacity question** — a confound to state rather than
to claim as a bonus.

**(b) Gate population — three tiers.**

| tier | slices | role |
|---|---|---|
| **gates** | TESS | 100% of the deployment population |
| **mandatory diagnostic, alarmed** | Kepler, K2 | a >0.02 AUC drop does not block promotion but **requires a written explanation in the roadmap first** |
| **reported, never gates** | all-mission | its weights are a sampling artefact |

The aggregate is unfit to decide anything: **Kepler is drawn at exactly
1,250/1,250 by construction**, so any all-mission number is weighted by a
sampling decision, and the 4,605-row comparison weighted **Kepler 48.6% / TESS
51.4% / K2 0%** in a decision whose real consequences are 100% TESS.

**This is not goalpost-moving, and here is the check that shows it:** on TESS
alone the branch model **still fails**, on recall @1% FPR — 0.238 against the
incumbent's 0.307. Narrowing the gate to the deployment slice does not rescue
run 1.

**Scope:** training and evaluation populations are **unchanged** — all three
missions, always, reported per mission. Only the promote/reject *decision rule*
narrows. Enforced in `validation/promotion.py`; summaries without a
`per_mission` block fall back to pooled means.

**(c) Recall @ 1% FPR is now a first-class gate criterion**, alongside AUC and
Brier. AUC scores ranking at every threshold; a follow-up shortlist lives at
exactly one. Current TESS numbers:

| | incumbent | branches |
|---|---:|---:|
| recall @1% FPR | **0.307** | **0.238** |
| recall @5% FPR | 0.561 | 0.550 |
| recall @10% FPR | 0.731 | 0.689 |
| Brier | 0.1211 | **0.1194** |

The branch model is better *calibrated* on TESS and worse where it is used.

**(d) Queued, not built: injection-recovery on matched hosts with observation
baseline held CONSTANT.** It is the only causal measure of detection
performance available here, and it is immune to the label-selection confound
that moved to stage 3. Build it when 2(b) is actually run.

### K2 was unbenchmarked for 9.7% of training — now it is not

The incumbent's `predictions.parquet` holds 4,818 out-of-fold rows and **zero
K2**: that run predates K2 in the catalogue. Every comparison against it
inner-joined all 527 K2 rows away silently.

Scored 2026-08-06 (`pipeline/scripts/evaluate.py score`,
`results/incumbent_k2_benchmark.json`):

| K2, n=527 | incumbent | branches |
|---|---:|---:|
| ROC-AUC | **0.9348** | 0.9189 |
| Brier | 0.1538 | **0.0957** |
| ECE | 0.1989 | **0.0500** |
| recall @1% FPR | **0.190** | 0.089 |

**Read it with its asymmetry.** No K2 row was in any of the incumbent's
training folds, so its numbers are **zero-shot cross-mission transfer**; the
branch model's are ordinary out-of-fold with K2 in four folds of five. Ranking
is comparable and the incumbent wins it by 0.0159 despite the handicap.
Calibration is not comparable — the incumbent's Platt scalers were fitted on
Kepler+TESS validation rows and K2's base rate is 0.598, which is most of the
0.1989 ECE.

Two rebuild details that would otherwise return confident wrong numbers: the
9-dim and 13-dim aux layouts **disagree at index 7** (catalogue SNR vs
`pink_snr`), so the vector is rebuilt rather than sliced; and catalogue SNR is
absent on **all 527** K2 rows, so that lane imputes.

`eval/comparison.py` now reports per-mission coverage whenever two prediction
sets are compared, and names any mission an inner join drops entirely. A mission
falling out of a comparison cannot be silent again.

### Stage 2(a) run 2 — the resolution fix, pre-registered 2026-08-06

**One change, both halves together: global 301 → 2001, local 31 → 201.** They
are the same mechanism testing the same hypothesis; splitting them costs two
runs to answer one question. Global-vs-local attribution is a follow-up
ablation and only matters if the fix works. Everything else is held:
`init_filters=16`, `conv_blocks=2`, same folds, same seed, plus the augmentation
built for this run.

**Sizing, measured before launch** (296-target probe at the new resolution;
peak RSS sampled during `fit()`, not read off at the end):

| | 301/31 | 2001/201 | |
|---|---:|---:|---|
| parameters | 192,817 | **233,617** | +21.2% |
| shards on disk | 122 MB | **~669 MB** | 126.4 KB/example, ×5.5 |
| interim per-target cache | 92 MB | ~360 MB | |
| peak training RSS | 4,861 MB | **~5.4 GB** | on 26 GB; ~4.7 GB is fixed TF/Metal cost, not dataset size |

**State the confound rather than bank it.** At 233,617 parameters the resolution
change lands *above* the incumbent's 227,641, and above the 226,711 the
cancelled capacity run (`init_filters=22`) would have reached. So this run
carries more capacity than the capacity experiment would have, and **a Kepler
gain cannot be attributed to resolution alone**. That is the sense in which it
partially subsumes the cancelled run — and it is a reason the pre-registered
reading below is about the *size* of the move, not its existence.

**How the result will be read** — fold std ~0.006, committed before the run
finishes:

| outcome | reading |
|---|---|
| Kepler gap closes to **under ~0.012** and TESS does not regress | resolution was the cause. Proceed to 2(b) |
| Kepler gap roughly **halves** | plausible, unproven. Report as-is, proceed |
| Kepler gap stays **above ~0.020** | resolution hypothesis **FALSIFIED**. The capacity run becomes mandatory under the trigger above. Say so and **stop; do not tune** |

**TESS must not regress.** TESS AUC *and* recall @1% FPR within noise of the
current branch model. A Kepler win bought with a TESS loss is a failure, not a
trade. K2 is reported alongside both — 9.7% of training, and benchmarked for
the first time on 2026-08-06.

**Second test, on the gating mission.** The 62 TESS targets with 300+ transits
caught carry a +0.0866 gap — the largest in the absolute-band table above. If
resolution is the cause, that band improves too. It is 62 rows, so it cannot
carry a decision on its own; it is recorded as a directional check that the
mechanism is not a Kepler-only story.

### Run 2 result — the resolution hypothesis is FALSIFIED (2026-08-07)

`models/cv/branches-20260807-2001`. Gate: **REJECT**. `ca906040` stays served.

**It made every slice worse, and roughly doubled the gap it was meant to close.**

| slice | incumbent | run 1 (301/31) | run 2 (2001/201) | run 2 gap |
|---|---:|---:|---:|---:|
| TESS *(gates)* | 0.9100 | 0.9079 | **0.8944** | **+0.0156** |
| Kepler | 0.9914 | 0.9566 | **0.9207** | **+0.0707** |
| all | 0.9558 | 0.9337 | **0.9043** | **+0.0516** |
| TESS recall @1% FPR | 0.307 | 0.238 | **0.126** | |

The pre-registered reading said a Kepler gap **above ~0.020 falsifies**. It went
from +0.0348 to **+0.0707**. The sharper cell-level test fails the same way:
Kepler 0–10 transits went from +0.0269 to **+0.2038**, so the damage is
concentrated where evidence is *thinnest* — the opposite of what a resolution
deficit predicts.

**Under the Task 2(a) trigger the capacity run is now MANDATORY**, and per the
pre-registration this stops here rather than tuning.

### What the run also uncovered: the noise floor was never measured

Fold 0 was re-run five times through `run_fold` with the trainer's own seeding,
one process each:

```
0.8927  0.8942  0.8984  0.9083  0.9179
mean 0.9023   sd 0.0106   range 0.0252
```

`set_global_seed`'s docstring already says it "doesn't make TF fully
deterministic on GPU", and nothing sets `enable_op_determinism` — so this is
known behaviour that had simply never been quantified. **Single-fold training
noise is sd ≈ 0.011.**

Two things follow, and they point in opposite directions:

- **It does not overturn the result.** The 5-fold mean averages this down —
  between 0.0048 (independent folds) and 0.0106 (fully correlated). Run 1 vs
  run 2 differ by 0.0313, which is **2.9σ to 6.6σ**. The falsification stands
  comfortably.
- **But the `±` in every `cv_summary.json` is not the run's uncertainty.** It is
  the spread *across folds within one run*, which mixes genuine fold-to-fold
  variation with training noise and says nothing about whether re-running the
  same configuration would reproduce the headline. Any future decision on a
  margin under ~0.02 needs repeat runs, not a single run's fold std.

**One knock-on, deferred deliberately.** The candidate view set
(`data/processed/candidates_viewset/`, 5,347 rows) is still at 301/31, so a
run-2 model cannot score candidates until it is rebuilt — about two hours. That
blocks the candidate-population bias measurement and stage 2(b)'s control arm,
but not run 2's own promote/reject decision, and it is wasted work if the
resolution hypothesis is falsified.

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
