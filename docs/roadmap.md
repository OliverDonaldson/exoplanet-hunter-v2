# Roadmap — the ExoMiner-inspired rebuild

Adopted 2026-07-26 after reviewing [NASA's ExoMiner](https://github.com/nasa/ExoMiner)
(ExoMiner++, TESS paper: [AJ 170, 5](https://iopscience.iop.org/article/10.3847/1538-3881/ae03a4)).
The UI redesign stays the locked final step.

We reimplement and credit; we do not vendor their code (NASA NOSA licence).

The single record of what was measured and what remains. Four conventions are
load-bearing: **pre-registration blocks are verbatim and never rewritten**, and
a result landing outside one is reported as falsified rather than re-specified;
**`W1`–`W14`** is the weakness register at 1d; **stage numbers were remapped
once**, on 2026-08-08, with the permanent mapping at 1c; and **nothing promotes
without being asked**.

## 1. Orientation

### 1a. Why ExoMiner

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
baseline* under stage 8. The transit-count row above is also restated: the
original **−0.003** was measured against `expected_transit_count`, the transits
the ephemeris *predicts*; against the transits actually *captured* it is
**−0.048**. The conclusion survives the correction, the number does not.

### 1b. What we take, and what we do not

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

### 1c. Stage numbering — the permanent old→new mapping

Stages were inserted as they were discovered, so the scheme grew into `0, 1, A,
2(a), 2(b), 2(c), C, D, 3, 2(d), G, 4, 5` — neither consecutive nor in execution
order. Renumbered 2026-08-08 to **flat consecutive integers in execution order**.

**This table is permanent.** Run directories on disk (`branches-20260807-shared`,
`branches-20260808-capacity`) and every commit message carry the old labels and
cannot be rewritten, so the mapping is the only thing that keeps them findable.

| new | old | stage |
|---|---|---|
| **1** | 0 | housekeeping, landmines |
| **2** | 1 | ExoMiner-grade inputs |
| **3** | A | re-baselined incumbent summary |
| **4** | 2(a) | per-diagnostic branches |
| **5** | C | leakage key + candidate rebuild |
| **6** | *(new)* | recall variance + re-baseline |
| **7** | D, absorbing 2(b) + 2(c) | branch attribution |
| **8** | 3 | labels and negatives |
| **9** | 2(d) | difference-image branch |
| **10** | G | Optuna re-tune |
| **11** | 4 | serving parity + explainability |
| **12** | 5 | UI redesign |

**Two conventions, so a reader can tell an edit from the record.** Headings and
prose cross-references are renumbered, with the old label in parentheses on first
mention. **Blocks that are the verbatim record of a pre-registration are not
renumbered** — the pre-commitments of 2026-08-06, the re-derived trigger, run 2's
"how the result will be read" table, and stage 2(b)'s re-specified criterion say
what was committed to at the time. Where one of those blocks points forward to a
stage, the new number is added in **square brackets** — `stage 3 [now 8]` — which
is an editorial insertion, not a rewrite.

### 1d. The weakness register — what `W1`–`W14` mean

Every weakness the project has measured in itself, ranked by damage to the
product's actual job: **ranking candidates for follow-up**. Each row is
measured from this repo's own artefacts. The `W` labels are referenced
throughout this file and in the handovers, which is the only reason they are
codes rather than sentences.

Merged here from `plan-2026-08-09.md` on 2026-08-14, with W1's status updated
for the stage 8 result and W13's for the frontend work having landed.

#### 1d.1 Tier 1 — defeats the product's purpose

| # | weakness | evidence | owner |
|---|---|---|---|
| **W1** | **Ranking is driven by observation baseline, and the signal is in the labels** | corr(baseline, label) **+0.278 all, +0.387 TESS**. TESS confirmed planets median 1,495 d vs 430 d for FPs | **stage 8 — done 2026-08-14, half delivered** |
| **W2** | **The model scores the star, not the transit** | **26.4%** of hosts pass with no injection — 46.7% planet hosts vs 12.3% FP hosts | **stage 7i** measured, **stage 8 did not move it**, **stage 9** attacks |
| **W3** | **No branch model has ever beaten the champion where it is used** — *as a replacement. As a complement it now does* | five arms rejected, all on shortlist recall: 0.238 / 0.126 / 0.145 / 0.236 / 0.220 against **0.307**. But the **ensemble** reaches **0.4362**, 3.9x its floor (3.11c) | **stage 10.5 answered it 2026-08-15**; stage 10, then a written decision |

W1 is the worst thing in the project. For the deployment use it is actively
counterproductive: it promotes targets that already received follow-up over
under-observed ones that may deserve it. No architecture can reach it.

**Updated 2026-08-14, after stage 8.**

- **W1 is half closed.** The original wording said the label correlation sat
  *"above every model"*. That is no longer true of arm P: propensity weighting
  put the branch model **below** its own labels (gap −0.0071 against +0.1265).
  **Target B — the architecture's amplification — is gone. Target A — the bias
  in the labels — is untouched and unreachable**, since the label correlation on
  a frozen evaluation slice is +0.3874 by definition. See 3.9b.
- **W2 survived stage 8 and is unchanged.** Prediction 4's *split* fell, but the
  threshold-free measure of the same construct did not move (host-AUC 0.6234 →
  0.6045, CIs overlapping, p≈0.33). Stage 9 remains its only instrument. See 3.9c.
- **W3 has a live route that is not stage 10.** Stage 10.5 asks whether the
  branch line is a *complement* rather than a replacement. A favourable answer
  reopens nothing about stage 4 — those rejections were about replacement.

#### 1d.2 Tier 2 — blocks delivery

| # | weakness | evidence | owner |
|---|---|---|---|
| **W4** | **A branch model cannot be scored from a light curve at all** | `TargetScorer` builds views with `preprocess.views`, not `preprocess.viewset`; `ScoringEnsemble.from_registry` loads `cnn_dualview.keras` | **stage 11** |
| **W5** | **No score can be explained** | per-branch contributions do not exist; the UI has nothing to display | **stage 11** |
| **W6** | **`score_std` is computed and thrown away** | not persisted, not in the catalogue, not surfaced per candidate — and it is a real differentiator (ExoMiner concedes theirs "is NOT a probability") | **finishing touches** |

#### 1d.3 Tier 3 — unexplained, and currently unowned

| # | weakness | evidence | owner |
|---|---|---|---|
| **W7** | **The narrow-span, high-count Kepler cell** | **+0.1446**, moved by **0.0002** across two bin resolutions, four fixed input defects, tied odd/even weights and a shared tower | **finishing touches** — decide or explain |
| **W8** | **The score does not track transit evidence** | corr(prob, transits caught) **−0.048**; the labels themselves sit at −0.073 | **stage 7ii** reports; no defensible target exists |
| **W9** | **TESS is the weakest mission and the only one served** | TESS **0.9100** vs Kepler 0.9915 | **stage 10** |

#### 1d.4 Tier 4 — engineering and operational risk

| # | weakness | evidence | severity |
|---|---|---|---|
| **W10** | **Repeated `run_cv` in one process slows without bound** | 86s → 108s → 161s → 190s on *identical* runs; one file did not finish in 3 h. `clear_session()` falsified as the fix; cause open | real hours lost, mitigated not fixed |
| **W11** | **Eager `GradientTape` over the assembled model aborts the process** | `Fatal Python error: Aborted` in `_ConcatGradV2`, TF 2.17.1 / Keras 3.15.0 on Metal. Reproduced on unmodified HEAD | **does not block stage 11** — see below |
| **W12** | ~~**No rate limiting on `/score`**~~ **CLOSED 2026-08-09** | each request triggers a network download + TF inference on a 2 GB box; public, unauthenticated | **fixed** — `api/app/ratelimit.py`, 12 tests |
| **W13** | **4 npm advisories (3 high) in build tooling** | `postcss` path traversal + 3 others; `npm audit fix` available | low real exposure — build-time only, the deployed console is static. **Unblocked 2026-08-14** — see below |
| **W14** | ~~**`HANDOVER.md` is 2,077 lines, superseded and partly wrong**~~ **CLOSED 2026-08-15** | carried old stage labels throughout; three documents pointed around it | **retired.** Its unique content — the DV and Gaia coverage measurements, and the merge collision that dropped the transit counts past all seven gates — was extracted into `data_provenance.md` first; the file itself is in git history |

**W11 does not block explainability, and this is worth stating because assuming
otherwise would re-scope stage 11 for no reason.** Branch-occlusion is *forward
passes* — mask a branch input, re-predict, take the difference. It never
constructs a gradient. Only gradient-based attribution (saliency, integrated
gradients) is blocked by the Metal abort, and that is not what stage 11 specifies.

**W13's blocker is gone — checked 2026-08-14.** `npm audit fix` was deferred
because `frontend/package-lock.json` carried another session's uncommitted
`animejs` addition, and running it would have entangled that work in a security
commit. `frontend/` is now clean and the lockfile committed, so this is the one
command it was always meant to be. Build-time only; **Ollie's call**.

**W10 is live for stage 10.5.** Repeated `run_cv` in one process slows without
bound, and 10.5 runs three CV passes. **One process per CV run**, without
exception.


## 2. Where the project stands

### 2a. Stage status — one table, kept current

One table, kept current. Detail for each row is in the stage sections below.

| stage | status | what closed it, or what is left |
|---|---|---|
| **1** *(old 0)* housekeeping, landmines | **done** | 71 GB staging reclaimed; two scripts that could silently write bad data deleted; four audit items fixed; TRICERATOPS vendored |
| **2** *(old 1)* ExoMiner-grade inputs | **done** 2026-08-05 | 5,423 examples × 11 branches, 3.6 GB DV archive, DV scalars, Gaia RUWE, FFI recovery, seventh gate |
| **3** *(old A)* re-baselined incumbent summary | **done** 2026-08-08; **its second half was not done until 2026-08-12** | `evaluate.py summarise` → `models/cv/incumbent-rebaselined/`. This row read "the gate returns decisions again instead of refusing" for four days and **that was false** — the summary existed but nothing routed the gate to it, so `promotion_gate.py` went on refusing every candidate on paperwork. `--incumbent-summary` closes it; see *The promotion gate was not calibrated* below. **Re-verified 2026-08-14**: it regenerates byte-identical, and the "the label change invalidates it" note carried against it was wrong on both counts — see the stage 8 result |
| **4** *(old 2(a))* per-diagnostic branches | runs 1, 2, 3 and the capacity arm all **REJECTED** — stage closed | run 3 reached the incumbent on TESS AUC (−0.0030, inside noise) and is better calibrated, but catches **less than half** as many planets at the shortlist threshold (0.145 vs 0.307). The capacity arm then falsified capacity as the cause: +19% params, paired d=−0.44 |
| **5** *(old C)* leakage key + candidate rebuild | **done** 2026-08-08 | `_cache_path` keyed on the ephemeris; candidate set rebuilt cold at 2001/201 — **5,346 rows, 309 MB, 95 min**. Run 3's checkpoint scores it, so the control arm and the candidate-bias measurement are unblocked |
| **6** recall variance + re-baseline | **done** 2026-08-09 | `pooled_gate_recall_seed_sd` **0.0292** → **a recall @1% FPR margin under ~0.034 is not a decision**. Run 3's rejection is sound at 4.8× the floor; the capacity arm's 0.145 → 0.236 was **real, not noise**, and stays unactionable. `models/cv/branches-20260808-rebaseline` is the control for every stage after it |
| **7i** *(old D)* offline control-arm harness | **done** 2026-08-12 — criterion **NOT met**; the branch line has no measured advantage on either criterion | the instrument: `clean`/`flatten` → `inject_box_transit` → `build_view_set` → `write_viewset_shards` → `make_viewset_dataset` → a run directory's fold members and calibrator. It is **stage 8's measuring instrument**, which is why it leads |
| **7ii** *(old D)* branch attribution | **deferred behind 8, 9 and 10** — 3-family sweep done 2026-08-09, **all arms null** | branch-drop mechanism built and declared in `run_config`. `unfolded`, `periodogram`, `scalar_only` all read null against the re-baseline; the one nominal PASS clears its bar by 0.23% and is an artefact of a 3-draw sd. Runs **once**, late, on a branch set and a distribution that have stopped moving |
| **8** *(old 3)* labels and negatives | **done** 2026-08-14 — four arms measured 2026-08-13, prediction 4 on 2026-08-14; **all four pre-registered predictions falsified** | **propensity weighting eliminated the architecture's amplification of the baseline confound at no measurable cost** (gap +0.1265 → −0.0071, 3.3× its bar). Synthetic negatives null; arm S unreadable by construction. The control-arm split also fell (−0.0966, 1.3× its bar) but **threshold-free host-scoring did not move**, so that second win is recorded as *qualified*. Group (a), external catalogue negatives, deliberately not done |
| **10.5** the ensemble arm | **CLOSED 2026-08-15 — BOTH ARMS CLEAR**; the control-arm pass landed the same day (3.11e) | **the branch line's value is as a complement, not a replacement.** Mean-of-logits recall @1% FPR **0.4362** (E-C) and **0.4223** (E-P) against the common-fold dual-view member's 0.3046 — **3.9x and 4.1x** their own floors. Reopens nothing about stage 4, whose rejections were about replacement. Nothing promotes |
| **9** *(old 2(d))* difference-image branch | not started | the only genuine *build* left in the model; needs the 11–17 px stamps re-gridded to a fixed size |
| **10** *(old G)* Optuna re-tune | not started | on the winner, after the distribution is settled |
| **11** *(old 4)* serving parity + explainability | not started | branch-occlusion contributions through `/score`; carries `score_std`, provenance headers, precision@k |
| **12** *(old 5)* UI redesign | locked last | |

Two rows the old table carried separately are gone by absorption, not by
cancellation: **old 2(b)** (unfolded-flux branch, rebuilt 2026-08-08) and **old
2(c)** (trend + periodogram, built) were never separate build steps — see the
audit finding below — so what is left of both is attribution, which is stage 7.

**Serving is unchanged throughout: `ca906040` (9-dim, 2001/201) on Fly.** Nothing
in stages 1–5 has been promoted, and the registry has not been touched since
2026-07-19.

### 2b. What the 2026-08-07 audit changed about this table

Three findings from reading the code rather than the plan. Each one invalidated a
row above.

**1. The unfolded-flux and trend/periodogram steps (old 2(b), 2(c)) were never
separate build steps.**
`build_cnn_branches` iterates `for name in VIEW_SHAPES` and builds a branch for
every one of the eleven views. Run 2's saved config carries
`unfolded_view_fc`, `trend_view_fc`, `periodogram_view_fc` and
`periodogram_masked_view_fc`. So **every branch model ever trained here carried
all eleven branches at once**, and the old stage 2's "each sub-step gated" design
was never the implementation path. Both stage 4 rejections were rejections of the
whole eleven-branch model. What remains for the two sub-steps is ablation — does
this branch earn its place — which is why they collapse into stage 7.

**2. The gate cannot engage against the incumbent.**
`_gate_slice` needs a `per_mission` block. `ca906040`'s summary has none, and the
re-baseline exists only as `results/incumbent_rebaselined.parquet` — there is no
re-baselined `cv_summary.json`, and `promotion_gate.py` has no flag to point at
one. Every branch-model decision from here is blocked until stage 3 produces it.
`per_mission_summary()` already exists in `train_branches.py`, so this is small
work, but it is on the critical path.

**The trap inside that fix.** The re-baselined parquet mixes two protocols, and
computing `per_mission` naively over it rebuilds the defect the audit just
closed:

| mission | out-of-fold | zero-shot | zero-shot base rate |
|---|---:|---:|---:|
| TESS *(gates)* | 2,371 | 0 | — |
| Kepler | 2,238 | 243 | **0.000 — pure negatives** |
| K2 | 0 | 527 | 0.598 |

TESS is 100% out-of-fold, so the gating slice is exact. Kepler is not: pooling
blends 2,238 out-of-fold rows with 243 zero-shot rows that carry **no positives
at all**.

**Measured, before assuming damage: the recorded Kepler figure survives.** Pooled
is **0.9915**, out-of-fold only is **0.9914** — a difference of **+0.0001**. AUC
is rank-based and the incumbent already ranks those 243 negatives low, so they
barely move it. No recorded number and no recorded decision changes.

That makes this *more* dangerous rather than less. It is a check that returns a
plausible answer which this time happens to be the right one — so it would never
have been caught by eye, and the next model, or the next mission split, has no
obligation to be so lucky. The summary builder must therefore compute
`per_mission` from out-of-fold rows only, carry K2 as a separately-labelled
zero-shot diagnostic, and **raise** if a slice spans more than one protocol.

*(The `n` differences between tables are join denominators, not errors: 2,371
TESS rows join `labels.parquet`, of which 2,367 survive the shared join against a
candidate's predictions. `evaluate.py compare` already reports the coverage and
names any mission the join drops.)*

**3. The candidate view set is at the wrong resolution.** *(Fixed 2026-08-08 —
see "The candidate view set, rebuilt" below.)*
`candidates_viewset/viewset.npz` was 301/31 against training's 2001/201, so no
post-run-1 model could score it. Resolution is baked into the npz, so the
five-minute `shard_viewset.py` path did not apply — this was the
`build_viewset.py` rebuild. It blocked the only surviving criterion of the
unfolded-flux step — now stage 7's — the candidate-population bias measurement,
and any candidate catalogue refresh.

**Sequencing consequence.** `_cache_path` keys on `(mission, tic)` with no planet
identifier. `labels.parquet` is 5,703 rows across 5,703 unique TICs with zero
duplicates, so the grouped-CV guard is currently degenerate and the cache
collision is dormant — but **a catalogue refresh is precisely what introduces
multi-planet hosts**. The key must be fixed before the next refresh, and before
the candidate rebuild, which is why stage 5 is ordered fix-then-rebuild.

### 2c. Uncovered, fixed, and improved since stage 2 closed

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
| **The training noise floor had never been measured**, and the `±` in every `cv_summary.json` was being read as the run's uncertainty when it is the spread across folds within one run | run 2's fold 0 not reproducing in a diagnostic | quantified: single-fold sd **0.0106** over 5 repeats; recorded under stage 4 |

| **The incumbent had never been scored on the current view set**, so every branch-model comparison ran against a 2026-07-19 baseline whose 4,818 rows predate K2, the DV scalars and the merge-collision fix | auditing which rows each `cv_summary.json` mean was computed over | `eval/scoring.py`, re-baselined set at `results/incumbent_rebaselined.parquet` |
| **The TESS gate never actually engaged.** `_gate_slice` returns `None` for a summary without a `per_mission` block, and the live incumbent has none — so every stage 4 decision silently fell back to comparing pooled means over different populations | regenerating the gate's decision from the real artefacts | `_population_mismatch` refuses the unmatched pooled comparison in `validation/promotion.py` |

| **Odd and even transits went through separate conv towers**, so the depth difference the head reads was partly a difference between two independently-learned sets of kernels — in the branch this model exists to make | reviewing ExoMiner's `build_joint_local_conv_branches` | one `TimeDistributed` tower over the flux family; fusion takes `odd - even` |
| A fold's score was a single seed draw, and the `±` in every summary mixed seed variance with fold difficulty with no way to separate them | the 0.0106 noise floor having no home in the artefact | `CVConfig.n_models_per_fold`; `summary.variance` reports `seed_sd` and `fold_sd` apart |

Tests: **304 → 428** (396 pipeline + 32 api). ruff and mypy clean on the
pre-commit config. Seven data gates pass.

### 2d. The shared flux tower — 2026-08-07

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

### 2e. Audit of the recorded numbers — 2026-08-07

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

### 2f. Execution order — the dependency graph, and why it is not the numbering

**Raised 2026-08-09: the roadmap is not sequential.** It is correct, and the
back-and-forth is real rather than cosmetic. Three backward edges existed in the
plan as written:

| edge | what it meant |
|---|---|
| **7 → 11** | stage 7's criterion needs a branch model scoreable from a light curve, which is stage 11's first half, four stages later |
| **8 → 7** | stage 8 changes the training distribution, so stage 7's attribution numbers are invalidated after they are measured |
| **9 → 7** | stage 9 adds a branch, so an attribution done at stage 7 describes a branch set that no longer exists |

Each was recorded honestly and none was ever resolved into an *order*. The 7 → 11
edge was worked around on 2026-08-09 by adopting the offline harness; the other
two were left as knock-ons to absorb later.

**Do not renumber a third time.** The numbers were already reassigned once
(2026-08-08) and have since become *names*: run directories, commit messages and
three documents refer to them, and `branches-20260809-drop-unfolded` would need
two hops to resolve under a third scheme. What was missing was never a numbering
scheme — it was a stated execution order. **The integers stay as stable
identities; the order below is what is executed, and it is the primary artefact.**

**One stage genuinely has parts, and splitting it removes every backward edge.**
Stage 7 owes two different things: an *instrument* (the offline control-arm
harness) and a *reading* (which branches earn their place). They have opposite
dependencies — the instrument blocks stage 8, the reading depends on stages 8, 9
and 10. Sub-steps as `i, ii` are the convention already set for exactly this
case.

| order | stage | depends on | why it sits here |
|---:|---|---|---|
| 1 | **7i** offline control-arm harness | nothing outstanding | it is **stage 8's measuring instrument**, not only stage 7's — pre-commitment (d)'s "injection-recovery on matched hosts with baseline held constant" is this harness. Also the only way to get a pre-stage-8 before-reading |
| 2 | **8** labels and negatives | 7i | the largest measured defect, and it invalidates everything measured before it — so it goes as early as its instrument allows |
| 3 | **9** difference-image branch | 8 | the last genuine build. After the distribution settles, or it is measured twice |
| 4 | **10** Optuna re-tune | 8, 9 | "on the winner, after the distribution is settled" — that is the settled architecture *and* the settled labels |
| 5 | **7ii** branch attribution | 8, 9, 10, 7i | attribution describes a **finished** branch set on a **settled** distribution. Run before any of them it is measuring something about to change, which is what the all-null sweep already spent six hours discovering |
| 6 | **11** serving parity + explainability | 7ii *(adjacency, not blocking)* | **stage 11's branch-occlusion and stage 7ii's leave-one-out are the same measurement at different granularity** — per-target against per-population. Running them adjacently validates the serving implementation against the population reading instead of leaving two independent attributions to disagree in public |
| 7 | **12** UI redesign | 11 | presentation only, locked last |

**No edge in that table points backwards.** Stages 1–6 are done and are not in
it. Stage 3's re-baselined incumbent summary is invalidated by stage 8 and needs
regenerating — that is a **repeat of a repeatable path**, not a backward edge,
and keeping stage 3 re-runnable rather than a one-off artefact is what makes it
so.

**The consequence for stage 7, stated plainly.** Its sweep has already run and is
all-null; per the sequencing assessment below, no further stage-7 CV compute is
bought before stage 8. Stage 7i finishes now, stage 7ii runs once, late, on a
branch set and a distribution that have stopped moving.

**One open item is on no stage at all.** The narrow-span, high-count Kepler cell
(+0.1446, unmoved by two bin resolutions, four fixed input defects, tied odd/even
weights and a shared tower) is described in this file as "the sharpest
unexplained thing in the model" and appears in no stage's contents. It needs an
owner or an explicit decision to leave it unexplained; it is currently neither.

### 2g. What stages 7–11 are worth — ranked by impact, 2026-08-09

Ranked against the product's actual job: **ranking candidates for follow-up**.
Not by build effort, and not by roadmap position.

| rank | stage | answers | impact if done | cost / confidence |
|---:|---|---|---|---|
| **1** | **8** labels and negatives | defect 5 | **The largest measured defect, and the only stage that can reach it.** Baseline correlates +0.278 with the label itself and +0.387 on TESS — *above every model*, so no architecture can touch it. For the deployment use it is actively counterproductive: it promotes targets that already received attention over under-observed ones that may deserve it. It improves **any** model, including the served champion | 25–35 h, **low** — external catalogue ingestion, whose only precedent was 5× out |
| **2** | **11** serving parity + explainability | delivery | **The only stage whose absence blocks shipping anything.** No branch model can be served at all until `TargetScorer` computes every branch live; `/score` returning per-branch contributions is what makes a shortlist justifiable per target rather than asserted; and stage 12 has nothing to display without it. Also carries `score_std`, provenance headers and precision@k | 10–15 h, medium. **No training compute** |
| **3** | **9** difference-image branch | defect 2 | The direct test of *"is this even the star we think it is"* — a centroid shift under the transit is how a background eclipsing binary is caught, and that is the host-scoring pathology at its source rather than at its symptom. The last genuine build in the model | 10–14 h, medium. Blocked on re-gridding 11–17 px stamps |
| **4** | **10** Optuna re-tune | defect 4 | Extracts what is left once architecture and distribution stop moving. Real but bounded — and it is the one stage that is almost entirely unattended, so it costs little attention | 12–15 h, medium-high, ~10–13 h of it unattended |
| **5** | **7** branch attribution | defects 1, 2, 3 | **Lowest as scoped, and the split is why.** 7i (the harness) is genuinely load-bearing — it is stage 8's instrument. 7ii (the reading) has already spent six hours returning four nulls, and leave-one-out structurally cannot separate redundancy from irrelevance. Its lasting deliverable is the instrument, not the attribution | 7i small; 7ii ~7 h compute, once, late |

**Read rank 5 correctly.** Stage 7 being last by impact is not an argument for
skipping it — attribution is what turns "eleven branches exist" into "these
branches earn their place", which is a claim the project should be able to make.
It is an argument for running it **once, at the end**, which is what the
execution order above does.

### 2h. Is Exoplanet Hunter ready once stage 11 is done?

Against the seven-point "what finished means" contract, restated in the table
below — **yes, with two named exceptions.**

| # | contract item | after 11 |
|---|---|---|
| 1 | a promotion decision made on evidence | **satisfiable** — see the caveat below |
| 2 | every number has an error bar | **done** — stage 6 delivered the recall floor; AUC had one already |
| 3 | control-arm host-pass rate moved off 26.4%, **or explained** | **satisfiable** — 7i measures it, 7ii and 9 are the interventions. "Explained" is an accepted finished state |
| 4 | ranking not driven by observation baseline | **stage 8's deliverable**, with the residual quantified rather than unknown |
| 5 | the score is a probability | **done and shipping**, plus `score_std` surfaced at stage 11 |
| 6 | every score can be explained | **stage 11's deliverable** — per-branch occlusion through `/score` |
| 7 | evaluation reproducible from artefacts | **already true**, and it has stayed true through an audit |

**Caveat on item 1, stated because it is the likely outcome rather than the
feared one.** Five arms have now been rejected — runs 1, 2, 3, the capacity arm
and the re-baseline — every one of them on shortlist recall. The probable
resolution of item 1 is **"the branch line is closed in writing and `ca906040`,
or its stage-10 retune, stays served"**, not a promotion. That is explicitly one
of the two finished states, and the handover already says so: only "we never
found out" fails. The apparatus that can tell those apart is itself a deliverable.

**Exception A — the narrow-span, high-count Kepler cell.** Unexplained across
three architectures, on no stage, and named in this file as the sharpest
unexplained thing in the model. It does not block shipping (Kepler is 0% of the
deployment population) but "ready" should not quietly include an unexplained
+0.1446.

**Exception B — distribution, which is not an engineering gap.** The one genuine
gap against ExoMiner is a published survey-scale catalogue with per-row
uncertainty, a DOI and a citation ask. It is a publishing task, it is deliberately
not on this roadmap, and stage 11 does not touch it.

**So: after stage 11 the product is complete and stage 12 is pure presentation**
— which is exactly the bar the contract sets. If the UI stage finds itself
needing a number the API cannot produce, a stage before it was not finished.


## 3. The record — what was measured, in the order it happened

Chronological. Pre-registrations sit immediately before the result they fixed
the reading of, so the order on the page is the order the work was done in.

### 3.1 Stages 1–3 — housekeeping, ExoMiner-grade inputs, the re-baselined summary

**Stage 1 *(old 0)* — housekeeping, landmines, vendoring.** *(done)*
71 GB of stitched-and-forgotten staging deleted and auto-cleanup added;
`preprocess_only.py` and `score_target.py` deleted (both could silently write
9-dim/no-K2 data or break on a non-9-dim model); the four remaining audit items
fixed (gate cwd, `dvc` resolution, MLflow run naming, the CI gate jobs
`ci.yml` had promised); patched TRICERATOPS vendored.

**Stage 2 *(old 1)* — ExoMiner-grade inputs.** *(done 2026-08-05)*
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
33x33**; that is Kepler's size, and stage 9 must re-grid.

Full detail, and the merge collision that silently dropped the transit counts
past all seven gates, in `data_provenance.md`.

**Stage 3 *(old A)* — the re-baselined incumbent summary.** *(done 2026-08-08)*
`evaluate.py summarise` writes `models/cv/incumbent-rebaselined/cv_summary.json`
with a `per_mission` block computed from out-of-fold rows only, so the gate
returns a decision instead of falling through to pooled means. Detail in "The
gate cannot engage against the incumbent" above, including the trap in the fix.

**Stage 4 *(old 2(a))* — the model, incrementally.** *(closed — all four arms
rejected)*
Originally specified as one stage with four gated sub-steps: (a) per-diagnostic
branches + scoped scalars + variance channels + joint local conv; (b)
unfolded-flux branch; (c) trend + periodogram branches; (d) difference-image
branch with quality attention, then an Optuna re-tune on the winner. **The
sub-step design was never the implementation path** — `build_cnn_branches` builds
all eleven branches at once — so what survives of it is: (a) is stage 4 and is
closed, (b) and (c) become attribution and are stage 7, (d) is still a genuine
build and is stage 9, and the re-tune is stage 10. Every arm passes the promotion
gate on CV AUC/Brier/ECE, the TESS slice, and injection-recovery completeness.


### 3.2 Stage 4 — per-diagnostic branches: three runs and a capacity arm

**Stage closed: every arm REJECTED**, all of them on shortlist recall.

#### 3.2a Run 1 — REJECTED (2026-08-05)

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
resolution test, and it is the Optuna step at the end of the old stage 2 — now
stage 10.

#### 3.2b Pre-commitments recorded before the next result exists

Written down first so they cannot be adjusted to fit an outcome.

*(Not renumbered — this block is the verbatim record of what was committed to on
2026-08-06. New stage numbers appear in square brackets where a reference would
otherwise be unresolvable; the mapping table is at the top of this file.)*

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

#### 3.2c The trigger, re-derived against run 3 — recorded 2026-08-08, before run 3 was read

The trigger fired legitimately and is **not** being cancelled. Its antecedent is
true even under the corrected comparison: re-baselined, run 2's Kepler gap is
**+0.0685** against a ~0.020 falsification threshold, and the re-baseline barely
moved the incumbent (0.9914 → 0.9915). The resolution hypothesis is properly
falsified.

What changed is not the trigger's validity but **its target**. `init_filters=22`
was specified against run 1's architecture, which carried the missingness
mission-indicator, two towers emitting `relu(bias)` on 56% of rows, a dead
`bootstrap_significance` and mission-blocked batches. None of those exist now.
Running it as written would answer a question about a model that has been
retired. So the obligation is discharged by asking the capacity question of the
*current* architecture, under a rule written before the result is known:

| run 3's Kepler gap | consequence |
|---|---|
| under **~0.012**, TESS not regressed | capacity is **not** the binding constraint. The arm is **redundant, not cancelled** — a 215,281-parameter model matched a 227,641-parameter incumbent, which is a stronger answer than the capacity run could have produced |
| **~0.020 or worse** | the capacity arm is **mandatory**, run as `init_filters=22` on run 3's architecture |
| in between | report as-is and decide in writing. **No default** |

The arm is specified as within-architecture (run 3 at `init_filters=16` vs `22`)
because cross-architecture parameter counts are not a clean capacity control —
the incumbent is a 9-dim dual-view CNN, not a branch model, so "215,281 below
227,641" is suggestive and not conclusive on its own.

**TESS still gates.** A Kepler reading cannot promote anything on its own, and
recall @1% FPR is read alongside AUC: run 2 sat within 0.016 of the incumbent on
TESS AUC while catching **less than half** as many real planets at the shortlist
threshold.

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
that moved to stage 3 [now 8]. Build it when 2(b) [now stage 7] is actually run.

#### 3.2d K2 was unbenchmarked for 9.7% of training — now it is not

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

#### 3.2e Run 2 — the resolution fix, pre-registered 2026-08-06

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
| Kepler gap closes to **under ~0.012** and TESS does not regress | resolution was the cause. Proceed to 2(b) [now stage 7] |
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

#### 3.2f Run 2 result — the resolution hypothesis is FALSIFIED (2026-08-07)

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

**Under the stage 4 trigger the capacity run is now MANDATORY**, and per the
pre-registration this stops here rather than tuning.

*(The trigger stands and the falsification holds — the Kepler gap is +0.0685 even
re-baselined. But `init_filters=22` was specified against run 1's architecture,
which no longer exists. The obligation was re-derived against run 3 on
2026-08-08, before run 3 was read: see "The trigger, re-derived against run 3"
above.)*

#### 3.2g Run 3 result — the fixed architecture on the fixed shards (2026-08-08)

`models/cv/branches-20260807-shared`. Gate: **REJECT**, on shortlist recall.
`ca906040` stays served; the registry is untouched.

The first run of the shared flux tower (215,281 params, *below* the incumbent's
227,641) on shards with the four defects fixed. Neither run 1 nor run 2 is a
baseline for it; the comparison is against the re-baselined incumbent, on the
5,375 rows both score:

| slice | n | incumbent | run 3 | gap | inc R@1% | run 3 R@1% |
|---|---:|---:|---:|---:|---:|---:|
| K2 | 527 | 0.9348 | 0.9028 | +0.0320 | 0.191 | 0.137 |
| Kepler | 2,481 | 0.9915 | 0.9464 | +0.0451 | 0.829 | 0.363 |
| **TESS** *(gates)* | 2,367 | 0.9100 | **0.9130** | **−0.0030** | **0.307** | **0.145** |
| all | 5,375 | 0.9523 | 0.9251 | +0.0272 | 0.439 | 0.263 |

**On TESS AUC it is the first branch model to reach the incumbent** — run 1
+0.0021, run 2 +0.0156, run 3 −0.0030 — and it is better calibrated on the
gating slice (Brier 0.1150 vs 0.1211, ECE 0.0171 vs 0.0438).

**And it is rejected anyway, on the criterion that matters.** TESS recall @1%
FPR is **0.145 against 0.307**: at the shortlist threshold it catches *less than
half* as many real planets. That is the same failure as runs 1 and 2 (0.238,
0.126) and it is what pre-commitment (c) exists to catch — a model that ranks
comparably overall and is worse exactly where it is used.

**The −0.0030 TESS win is not a win.** See the variance decomposition below: a
margin under ~0.009 is inside the noise. Level is the honest reading.

#### 3.2h The variance decomposition, measured for the first time

`--n-models-per-fold 3` makes `summary.variance` report the two components apart:

```
seed_sd 0.0081   fold_sd 0.0094   n_models_per_fold 3
```

- **`seed_sd` = 0.0081** is per-*model* training noise. It independently
  confirms the 0.0106 floor measured by re-running fold 0 five times in five
  processes — a different measurement path, arriving slightly lower.
- **`fold_sd` = 0.0094** is fold difficulty, a property of the split.
- They are **nearly equal**, so the ±0.0106 this project has been quoting was
  never "the run's uncertainty": it was roughly `sqrt(seed² + fold²)` with the
  two halves indistinguishable.
- Run 3 averages 3 models per fold, so the run-level reseeding sd is about
  `seed_sd/√3 ≈ 0.0047`. **A margin under ~0.009 is not a decision.**

#### 3.2i The capacity arm — trigger fired, launched 2026-08-08

Kepler is **+0.0451**, well past the ~0.020 threshold in the re-derived trigger
above, so the arm is **mandatory** and running as
`models/cv/branches-20260808-capacity` (`--init-filters 22`).

**One correction to the pre-registration, recorded rather than absorbed.** The
original prediction called `init_filters=22` "226,711 parameters, 0.4% from the
incumbent's 227,641" — computed on the *four-tower* architecture. On the shared
tower it is **256,711**, so this is a **+19% capacity test rather than the
incumbent-parity test it was designed as**. That makes the falsification
stronger, not weaker: if 19% more capacity does not close a +0.045 Kepler gap,
capacity is not the binding constraint.

#### 3.2j Capacity arm result — capacity is NOT the constraint (2026-08-08)

`models/cv/branches-20260808-capacity`, `init_filters=22`, 256,711 params
(+19% on run 3). Gate: **REJECT**. `ca906040` stays served.

**The clean test, paired fold by fold against run 3** — same split, same seed,
same shards, so unlike run 1 vs run 2 the pairing here is exact:

```
mean -0.0035, won 3/5, d=-0.44
```

Against a run-level reseeding sd of ~0.0047, that is **nothing**. On the shared
rows, +19% capacity moved the Kepler gap the wrong way:

| slice | incumbent | run 3 (16f, 215k) | capacity (22f, 257k) |
|---|---:|---:|---:|
| **TESS** *(gates)* | 0.9100 | **0.9130** | 0.9089 |
| Kepler | 0.9915 | 0.9464 (+0.0451) | 0.9449 (**+0.0466**) |
| K2 | 0.9348 | 0.9028 | 0.8997 |
| TESS recall @1% FPR | **0.307** | 0.145 | **0.236** |

**The trigger is discharged, and the cancelled prediction was right.** The
2026-08-06 pre-commitment recorded that `init_filters=22` "would move the Kepler
gap by less than 0.005". Measured: **−0.0015, in the wrong direction.** The
difference is that it is now tested rather than asserted, which is exactly what
the trigger existed to force. **Capacity is closed as a hypothesis.**

**One observation, deliberately not upgraded to a finding.** TESS recall @1% FPR
went **0.145 → 0.236** while AUC fell — capacity traded ranking for shortlist
performance. That is a large relative move on the criterion that has rejected
every run so far, but it is one run on a single-threshold statistic with no
variance estimate, and this project has been burned by exactly that shape of
evidence. It is worth a `--n-models-per-fold` repeat aimed at recall
specifically before anyone builds on it.

Also of note: `fold_sd` rose 0.0094 → 0.0147 while `seed_sd` held at 0.0082.
The extra capacity made folds diverge without making individual draws noisier.

#### 3.2k Three training-path changes that break comparability going forward

Recorded here because this project's recurring injury is a comparison that is
not like-for-like and does not say so.

**Augmentation masking is now gated by view kind (2026-08-08).** It previously
applied to all eleven views. Zero is the out-of-transit baseline on folded flux,
so masking there removes a measurement — but zero in `gap_view` asserts *no
cadence was missing*, and zero in a peak-normalised periodogram asserts *no
power at this period*, both while the presence channel still reports the bin as
measured. Three of eleven views were being fed a confident false claim on
`mask_prob` of their bins every epoch.

**Run 3 and the capacity arm are unaffected** — both had the module loaded
before the change — so the comparison that decides the capacity question is
still internally consistent. **Every run after them is not comparable to run 3
on this axis** and needs its own baseline. Stage 6 produces exactly that
re-baseline, and stage 7 re-baselines again anyway.

**The unfolded branch was rebuilt (2026-08-08).** Audit finding #23: it
convolved along the transit axis with the 201 phase bins flattened into 603
unordered channels, so it never saw a transit. It now runs a per-transit conv
tower under `TimeDistributed` and pools with a masked mean + max + spread. The
model drops **215,281 → 169,361 parameters (−21.3%)**, almost all of it the
48,256-parameter convolution that was doing the damage.

Runs 1, 2, 3 and the capacity arm all carried the broken branch, so **stage 4's
rejections stand** — nothing about this changes what those runs measured,
and a branch that could not see a transit is one more reason they lost. But no
run so far is a baseline for the rebuilt model. The direction is the opposite of
the capacity arm's (+19%, paired d = −0.44, nothing), which is weak evidence
that −21.3% is not decisive on its own. Weak evidence, not a measurement.

**The inner validation split is now stratified (2026-08-08).** It was
`GroupShuffleSplit`, which is group-aware but guarantees nothing about class
balance — and the Platt fit downstream requires both classes, or it converges
happily on a scaler that maps every score to one end. Both affected sites now
call `training/splits.py`'s `stratified_inner_split`; the random-forest holdout
at `train.py:132` is a test split and is untouched.

This changes the inner partition and therefore the numbers, which is why it had
been deferred to "between experiments". It landed here because the unfolded
rebuild already forces a fresh baseline — **one re-baseline absorbs both changes
instead of two**. Production was never at risk (~868 validation rows); it was
the tiny test fixtures that exposed it.

#### 3.2l The one cell three architectures have not moved

The narrow-span, high-count Kepler cell is unchanged across every run:

| | run 1 (301/31) | run 3 (shared, 2001/201) |
|---|---:|---:|
| span <4 bins, 100+ transits (n≈150) | **+0.1446** | **+0.1448** |

Two bin resolutions, four fixed input defects, tied odd/even weights and a
shared tower move it by **0.0002**. Whatever drives that cell is not resolution
(run 2 made it worse), not the missingness indicator, and not the separate
towers. It is the sharpest unexplained thing in the model and it deserves its
own investigation rather than another architecture pass.

#### 3.2m What the run also uncovered: the noise floor was never measured

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

  *(The ~0.02 was a conservative guess made while the two variance components
  were still inseparable. **Superseded 2026-08-08 by the measured figure: a
  margin under ~0.009 is not a decision** — see "The variance decomposition,
  measured for the first time" above, where `--n-models-per-fold 3` finally
  split `seed_sd 0.0081` from `fold_sd 0.0094`. That threshold is **AUC only**;
  recall @1% FPR still has no variance estimate, which is precisely why the
  capacity arm's recall jump cannot be acted on as it stands. **That gap is
  stage 6**, below.)*

**One knock-on, deferred deliberately.** The candidate view set
(`data/processed/candidates_viewset/`, 5,347 rows) is still at 301/31, so a
run-2 model cannot score candidates until it is rebuilt — about two hours. That
blocks the candidate-population bias measurement and stage 7's control arm,
but not run 2's own promote/reject decision, and it is wasted work if the
resolution hypothesis is falsified.

*(Superseded 2026-08-08: the two-hour figure assumed a cold cache and was then
overtaken twice. `data/interim/viewset/g2001l201` had accumulated **5,426
targets, 309 MB** at 2001/201, which would have made the rebuild mostly cache
hits — but the `_cache_path` ephemeris key renames every entry, so none of them
can be found and the build re-derives from the light curves. Budget a cold
rebuild of all 5,347 candidates.

Two orphaned caches, for two different reasons — measured 2026-08-08, do not
conflate them:

| path | targets | size | resolution | orphaned by |
|---|---:|---:|---|---|
| `data/interim/viewset/g2001l201/` | 5,426 | 309 MB | 2001/201 | the `_cache_path` ephemeris key — **a cost this change introduced** |
| `data/interim/viewset/*.npz` (loose) | 5,423 | 92 MB | 301/31 | superseded resolution, already dead before the key change |

401 MB total is reclaimable, but only the 309 MB is attributable to the
ephemeris key. Both are under the old `{mission}_{tic}.npz` naming and nothing
reads either.)*

**Both deleted 2026-08-08: 10,849 files, 395.9 MB** (312.5 + 83.4; the 401 MB
above was MB-vs-MiB, not a different set). `data/interim/` is gitignored and
carries no DVC pointer, so this is derived data with no artefact behind it.

**The delete had a trap in it worth recording.** By then `g2001l201/` held *two*
generations side by side — the 5,426 orphaned training-target entries under
`{mission}_{tic}.npz`, and the **5,346 ephemeris-keyed candidate entries the
rebuild had just written**, 326.8 MB of them. A glob over the directory would
have taken both. The two were separated by regex, asserted to be disjoint before
anything was unlinked, and `_cache_path` was then re-run over 3,000 catalogue
rows to confirm the survivors are still addressable: **2,995 hits, 5 misses.**

Note the asymmetry this leaves: what survives covers **candidates**, not
training targets. A future *training* view set rebuild is still cold — it was
already, since those entries were unfindable — so the cache on disk now helps a
candidate re-run and not a training one.

### 3.3 Stage 5 — the candidate view set, rebuilt (2026-08-08)

Done, cold, as budgeted: **5,346 rows at 2001/201, 309 MB, 95 minutes** for
7,174 catalogue rows. `ViewSetArrays.validate()` reports it well-formed, every
view matches training's `VIEW_SHAPES`, and **run 3's fold-0 checkpoint scores it**
— which is the whole point, since no post-run-1 model could touch the old set.
The 401 MB of orphaned cache was left in place; nothing reads it, and deleting
it is not this task.

Every one of the 7,174 rows is accounted for: 5,346 built, 1,803 no FITS, 17
preprocess errors, 8 with no ephemeris. Sources are 3,929 SPOC 2-minute, 719
FFI, 698 Kepler; DV usable on 64%, RUWE on 85%.

**The row count moved 5,347 → 5,346, and the three rows reconcile exactly
against the refresh.** TIC 60520371 and TIC 160476088 were dispositioned **FP**
on 2026-08-08 and so left `candidates.parquet` for `labels.parquet` — they are
precisely the two rows behind the catalogue's 5,703 → 5,705 growth. TIC
443534757 is a new **PC**. Net −2 +1.

Stated rather than assumed, because a candidate set that quietly changed size is
indistinguishable from one built over a population nobody chose — and per the
handover a rebuilt set **will** trip the gate's row-count alarm against run 3
and the capacity arm, which were measured on the previous catalogue. That alarm
is correct and should not be silenced.

**Stage 7's success criterion — recorded as stage 2(b)'s, re-specified
2026-08-05, and not renumbered inside.** It read
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
  — see stage 3 [now 8].
- **The transit-count correlation is reported, not gated.** Its zero point is
  **−0.048** against transits captured, not the −0.003 that was measured against
  transits predicted; and the labels themselves sit at −0.073, so there is no
  defensible target value to demand.

The clean test of the unfolded branch is **injection-recovery on matched hosts
with observation baseline held constant**, which removes the label confound
entirely. Build that harness when 2(b) [now stage 7] is run.

**Stage 6 — recall variance + re-baseline.** *(next)*
`recall @1% FPR` is the criterion that has rejected all four arms of stage 4 —
run 3 on **0.145 vs 0.307** — and it has **no variance estimate at all**. AUC's
noise floor was measured (`seed_sd 0.0081`, `fold_sd 0.0094`) and "a margin under
~0.009 is not a decision" adopted from it; the statistic that does the actual
rejecting never got the same treatment.

`_variance_decomposition` reads only `model_roc_auc`, and the per-member metrics
recorded beside it come from `classification_metrics`, whose `.recall` is recall
at threshold 0.5 — **not** the gate's statistic. The gate's `recall_at_1pct_fpr`
lives in `SliceMetrics` (`eval/comparison.py`), verified exact against sklearn
and a brute-force threshold sweep. So each member records its own
`recall_at_1pct_fpr` alongside its AUC, and the summary reports `recall_seed_sd`
/ `recall_fold_sd` beside the existing pair. **Purely additive to
`cv_summary.json`** — the promotion gate reads named keys and is unaffected.

**Three estimates, not one, because a fold is the wrong population.** TESS holds
2,399 rows at a 0.552 base rate, so a fold's TESS test slice carries ~215
negatives and its 1% FPR cut is **two rows** — the statistic is set by where the
third-highest-scoring negative lands, and its spread says more about that than
about the model. The gate reads the *pooled* out-of-fold set: ~1,074 negatives,
a cut of ten. So `predictions.parquet` gained one uncalibrated score column per
ensemble member, and stacking member *i*'s column across folds re-forms a
complete out-of-fold prediction set for that member alone. Three members, three
independent draws of **the number the gate actually reads**, at the cost of three
float columns and no extra training or inference.

| reported | what it is |
|---|---|
| `recall_seed_sd` / `recall_fold_sd` | the whole fold, every mission — mirrors the AUC pair exactly |
| `gate_recall_seed_sd` / `gate_recall_fold_sd` | that fold's TESS rows alone; **coarse, and an upper bound** |
| `pooled_gate_recall_seed_sd` | spread of the three pooled-TESS draws — **the primary estimate** |

Then the re-baseline itself: HEAD, `--n-models-per-fold 3`, 5 folds, over the
**unchanged** training shards (5,426 rows: 2,500 Kepler, 2,399 TESS, 527 K2).
Same rows as run 3 and the capacity arm, so the only differences are the three
code changes and the gate's row-count alarm should stay silent. It is the
control for every subsequent stage — three training-path changes have landed
since run 3, so nothing measured before 2026-08-08 is a baseline — and it returns
the recall noise floor for free.

**This replaces the queued capacity repeat**, which was specified against the
capacity arm's architecture. That architecture no longer exists on HEAD, so
running it now would measure the rebuilt unfolded branch rather than the capacity
arm — the same trap as `init_filters=22` being re-derived against run 3. This
measures the thing underneath instead, on the architecture that exists, for one
run instead of three.


### 3.4 Stage 6 — the recall noise floor

#### 3.4a Pre-registered before the run — recorded 2026-08-08, run not yet launched

Nobody is watching an autonomous session read its own result, so every number
below is written down first.

**The decision rule, taken from the AUC precedent rather than invented here.**
The adopted AUC threshold is `2 x seed_sd / sqrt(n_models_per_fold)`:
`2 x 0.0081 / sqrt(3) = 0.0094`, which is the recorded "a margin under ~0.009 is
not a decision". Applied unchanged to recall, a margin `M` **is not a decision**
when

```
per-member seed_sd  >=  M x sqrt(3) / 2
```

**The two numbers, computed before the run exists.**

| question | margin | the rejection/effect survives only if per-member `seed_sd` is |
|---|---:|---:|
| Is run 3's rejection sound? *(TESS R@1%FPR 0.145 vs incumbent 0.307)* | 0.162 | **below 0.1403** |
| Was the capacity arm's 0.145 -> 0.236 noise? | 0.091 | at or **above 0.0788** for it to be noise |

`pooled_gate_recall_seed_sd` is the estimator these are read against.
`gate_recall_seed_sd` is reported beside it and is an upper bound, so a
conclusion that survives *it* is safe a fortiori.

**How each outcome reads — fixed now.**

| `pooled_gate_recall_seed_sd` | reading |
|---|---|
| **< 0.0788** | run 3's rejection is sound, and the capacity arm's recall jump is **outside** reseeding noise. It stays **unactionable** regardless — the architecture it was measured on does not exist on HEAD — so it is recorded as a lead for stage 7, not built on |
| **0.0788 – 0.1403** | run 3's rejection stands; the capacity jump is **inside** noise and is retired as an observation. This fully discharges the queued capacity repeat |
| **>= 0.1403** | **run 3's margin does not support its own rejection.** Report as falsified, stop, and surface it. Do not re-specify the criterion. This does not un-reject run 3 — AUC, Brier and the gate's other guards are untouched — but it would mean the criterion this project has leaned on cannot carry the weight put on it, which is a finding |

**Prediction, recorded so it can be wrong.** `pooled_gate_recall_seed_sd` lands
in **0.02–0.08**, and `gate_recall_seed_sd` lands above it. The re-baseline's own
`per_mission.TESS` AUC lands **within ±0.009 of run 3's 0.9119**, and its TESS
recall @1% FPR lands in **0.10–0.30** — a deliberately wide band, because a
narrow one on the statistic whose spread has never been measured would be false
precision.

**Four caveats, recorded now rather than discovered later.**

1. **Reseeding noise is necessary, not sufficient.** 0.145 vs 0.307 compares two
   *different models*, not two seeds of one; the incumbent carries its own noise
   and a different architecture and protocol. Clearing the threshold does not by
   itself make the rejection sound. Failing it does make it unsound.
2. **Three draws is a thin sd.** The sampling spread of an sd from three draws is
   roughly 40% of its own value. If the result lands within a factor of ~1.5 of a
   threshold, the honest reading is **unresolved** — that is a stop-and-ask, not
   a re-specification.
3. **Two TESS populations, as always.** The pooled draws are over the run's own
   `per_mission.TESS`, **n=2,399**. The 0.145 and 0.307 in the table above are
   the **n=2,367** shared-join gate slice. Both are correct; neither is being
   "fixed" to match the other.
4. **The floor is measured on HEAD, applied to run 3's margin.** That assumes the
   noise scale is a property of the statistic and the population rather than of
   the specific architecture. Both are eleven-branch models on the same shards,
   so it is reasonable — and it is an assumption, not a measurement.

**What will be run, named before it exists.** One CV run to
`models/cv/branches-20260808-rebaseline`, `--n-models-per-fold 3`, 5 folds, over
`data/processed/viewset_tfrecords` unchanged. Then, as a **reported diagnostic
and not a decision**, `evaluate_promotion` against
`models/cv/incumbent-rebaselined/cv_summary.json` — the same protocol run 3's
table used, so the two are comparable. `promotion_gate.py` itself is not used:
it reads the registry, the live incumbent has no `per_mission` block and it has
no `--incumbent` flag.

**The tree condition, stated precisely rather than as "clean".** No *tracked*
file will differ from the recorded `git_sha` — `git status --porcelain
--untracked-files=no` is empty at launch. `run_config.git_dirty` will
nevertheless be **True**, because `--porcelain` counts untracked files and
`docs/demo-script-2026-08-08.md` is one: written by another session at 21:10,
read by nothing, and not this session's to commit. Recorded here so the flag on
the control run is explained by the artefact rather than guessed at later. Run 3
recorded no provenance at all and the capacity arm recorded `dirty: True` with
no explanation; this is the first branch run whose flag means something specific.

*(Outcome: the run recorded `git_dirty=False`, not True. The stated reason for
predicting True stopped holding between writing this and launching — a broad
`git add docs/` in `6005506` swept that untracked file into the commit, so the
tree really was clean. The provenance claim is stronger than pre-registered, but
it is stronger by accident, and the prediction was wrong for a reason worth
recording rather than quietly benefiting from.)*

**Nothing promotes.** The re-baseline is a control. `models/registry.json` is
untouched, `ca906040` stays served, and a favourable number here does not reopen
stage 4 — only stage 7's leave-one-out runs, read against this re-baseline, can
move anything.

#### 3.4b First launch failed on fold 0; the pre-registration is unchanged — 2026-08-09

The run died ~15 minutes in, in `fit_platt`, on the convergence guard added
2026-08-08. The guard was right: `nll` clipped `p` while the analytic gradient
did not, so BFGS's line search failed on an inconsistency between `f` and `jac`.
Fixed by the stable `softplus(z) - y·z` form; the guard is untouched. Full
detail in the audit.

**It implied nothing for run 3, and that was checked rather than assumed.** It
was recorded here first that runs predating the guard had "calibrated on a
stalled fit" — an inference from the guard's absence, not a measurement.
Regenerating run 3's calibration from its own checkpoints reproduces its stored
scores at **max |delta| = 0.0**, and refitting with the converged optimiser
returns the same parameters: **run 3's Brier and ECE stand exactly as recorded**
(TESS 0.1150 / 0.0171). The defect needs validation rows saturating past `_EPS`,
and run 3 had **2 in 4,344** against a score of exactly 1.0 in the fold that
stalled. See the audit for the per-fold table and for what remains unmeasured.

**None of the pre-registered numbers move, and this is why.** Every quantity the
thresholds above are read against is **rank-based**, and Platt is monotone:
`pooled_gate_recall` is computed from the *uncalibrated* `member_score_*`
columns and never touches a calibrator at all, and per-mission ROC-AUC and
recall @1% FPR are invariant to any monotone rescaling. The calibration fix
moves **Brier and ECE only**. The 0.1403 and 0.0788 thresholds, the three-way
outcome table and the predictions stand exactly as written before the first
launch — they are not being re-specified after a failure, and nothing about the
failure touched a result.

If the re-baseline's TESS AUC lands outside ±0.009 of 0.9119 in either direction,
that is a **falsified prediction** and is reported as one. It is not re-read as
an improvement or a regression: one run of a control cannot carry that, and stage
7 re-baselines again anyway.

#### 3.4c Result — the recall noise floor, measured (2026-08-09)

`models/cv/branches-20260808-rebaseline`, HEAD `6005506`, `git_dirty=False`,
5,426 rows, `--n-models-per-fold 3`. **Every pre-registered prediction landed
within its band**, and the outcome is **branch 1** of the three fixed before the
first launch.

**The deliverable.** Recall @1% FPR now has the error bar AUC has had since
2026-08-08:

```
pooled gate draws (n=2,399 TESS rows):  0.1857  0.1842  0.2355
pooled_gate_recall_seed_sd = 0.0292          <- the primary estimate
```

| floor | seed_sd | fold_sd |
|---|---:|---:|
| AUC *(existing; run 3 had 0.0081 / 0.0094)* | 0.0060 | 0.0101 |
| recall, all missions | 0.0485 | 0.0547 |
| recall, gate slice, **per fold** | 0.0689 | 0.0888 |
| recall, gate slice, **pooled** | **0.0292** | — |

The fold-level estimate is **2.4× the pooled one**, exactly as pre-registered —
a fold's TESS slice carries ~215 negatives and a 1% FPR cut of two rows, so it
was only ever an upper bound. Using it would have overstated the noise by more
than a factor of two.

**The threshold this yields**, by the same rule AUC's came from
(`2 × seed_sd / √n_models`):

> **A recall @1% FPR margin under ~0.034 is not a decision.**

**The two pre-registered questions, answered.**

| question | margin | vs 0.0337 | verdict |
|---|---:|---|---|
| Is run 3's 0.145-vs-0.307 rejection sound? | 0.162 | **4.8×** the floor | **sound** |
| Was the capacity arm's 0.145 → 0.236 noise? | 0.091 | **2.7×** the floor | **not noise** |

The capacity arm's recall jump was a **real effect**, not the artefact this
project feared. Per the pre-registration it stays **unactionable regardless** —
`init_filters=22` was measured on an architecture that is not on HEAD — so it is
recorded as a lead for stage 7 and nothing is built on it.

**The re-baseline itself** *(its own `per_mission` block, n=2,399 on TESS — not
the n=2,367 gate slice)*:

| slice | n | AUC | Brier | ECE | R@1% FPR |
|---|---:|---:|---:|---:|---:|
| K2 | 527 | 0.9351 | 0.1000 | 0.1141 | 0.1683 |
| Kepler | 2,500 | 0.9547 | 0.0839 | 0.0275 | 0.5328 |
| **TESS** *(gates)* | 2,399 | **0.9202** | 0.1108 | **0.0159** | **0.2196** |
| all | 5,426 | 0.9365 | 0.0973 | 0.0108 | 0.2855 |

Against the two runs on the identical shard set:

| | TESS AUC | TESS R@1% FPR |
|---|---:|---:|
| run 3 | 0.9119 | 0.1434 |
| capacity arm | 0.9076 | 0.2332 |
| **re-baseline** | **0.9202** | **0.2196** |

**Shortlist recall rose 0.1434 → 0.2196 against run 3 — a margin of 0.0762, or
2.3× the floor measured in the same run.** Read with pre-registered caveat 1:
reseeding noise is *necessary, not sufficient* for a cross-run margin, and this
one spans three training-path changes at once (unfolded rebuild, stratified inner
split, augmentation masking), so it attributes to none of them. It is also still
well below the incumbent's 0.307.

**Gate: REJECT**, run as a reported diagnostic and not a decision, exactly as
pre-registered. TESS AUC 0.9202 vs 0.9100 and ECE 0.0159 vs 0.0438 both favour
the candidate; **shortlist recall 0.220 vs 0.307 rejects it**, which is the same
criterion that closed stage 4 and it is now the first time that rejection is
backed by a measured floor. `ca906040` stays served; the registry is untouched.

**The Kepler alarm, explained as the pre-commitment requires.** Kepler fell
−0.0367 against the incumbent, past the 0.02 alarm. It is not a regression: at
0.9547 this is the **best Kepler any branch model has reached** (run 3 0.9464,
capacity 0.9449), so the gap narrowed rather than widened. The branch line has
lost on Kepler since run 1 and the cause is unresolved — the narrow-span,
high-count cell that three architectures have not moved.

**Two readings that are not decisions, recorded because they will be tempting.**
The +0.0083 TESS AUC over run 3 sits at 92% of its own pre-registered band and
just under the AUC floor's 2σ (~0.007 at this run's `seed_sd 0.0060`) — **level
is the honest reading**, not an improvement. And observation baseline sensitivity
rose to **+0.3025** (run 3: +0.2901), now *above* the +0.278 label correlation
itself; transit sensitivity moved further from zero at **−0.1467** (run 3:
−0.1249). Both are reported diagnostics, both point at stage 8, and neither gates
anything.


### 3.5 Stage 7 — the attribution sweep, and the criterion problem

#### 3.5a Stage 7's criterion is blocked on stage 11 — found 2026-08-09

**The control-arm host-pass rate cannot be measured for any branch model today,
and nothing recorded this.** `roadmap.md` gates stage 7 on "26.4% must fall",
measured on real hosts with no injection. That number comes from
`injection_recovery.py`, whose control arm is `snr_target == 0, depth 0` pushed
through the **live scoring path**. That path cannot carry a branch model:

- `ScoringEnsemble.from_registry` loads `cnn_dualview.keras` per fold and
  predicts from `global_view`, `local_view`, `aux_features`.
- `TargetScorer` builds those with `preprocess.views.build_views` — the
  dual-view builder, not `preprocess.viewset`'s eleven.
- The runner constructs `TargetScorer(models_dir=Path("models"))` and reports
  `scorer.ensemble.run_id`: it scores **whatever the registry serves**, with no
  flag for a CV run directory. Pointing it at an ablation by editing
  `models/registry.json` is a hard non-negotiable.

Adding a run-directory flag is necessary and nowhere near sufficient — the
serving path would still build the wrong views. **Making a branch model scoreable
from a light curve is stage 11's "`TargetScorer` computes every branch live"**,
budgeted 10–15 h and scheduled four stages later. Stage 7 as written depends on
it.

Separately, `select_hosts` filters on mission, label, depth and cache only:
**the baseline-matched harness owed since old 2(b) does not exist**, so even with
serving parity the hosts would not be matched on observation baseline.

Three ways out were put to Ollie: pull stage 11's serving parity forward; build
an **offline** injection/control harness; or re-scope and defer.

#### 3.5b Decided 2026-08-09: the offline harness, with the incumbent re-measured on it

**Why not pull serving parity forward.** Stage 8 already invalidates stage 7's
attribution numbers — that is why stage 8 was moved ahead of stage 9 — so
spending 10–15 h on serving parity to obtain a stage-7 number stage 8 obsoletes
is paying for it twice, the same argument in the same direction. And serving
parity earns its keep in stage 11 by *shipping*; pulled forward it does the same
work under another stage's pressure and ships nothing. What attribution actually
needs is cross-ablation comparability on **one fixed protocol**, which the
offline harness gives directly.

**It is cheaper than it first looked, because the chain is existing parts.**
`build_view_set(lc, *, period, t0, duration, trend_lc=None, raw_lc=None)` builds
all eleven views from a light curve and an ephemeris with **no DV products**. So:
`clean_lightcurve` / `flatten_lightcurve` (already composed in `host_baseline`) →
`inject_box_transit` (model-independent, already tested) → `build_view_set` →
`write_viewset_shards` → `make_viewset_dataset` → the run directory's fold
members and calibrator, loaded exactly as `train_branches.py` does. New code is a
baseline matcher, a driver and the pass-rate report: one script, one test file.
Compute is forward passes over cached FITS — trivial beside a 2 h CV run.

Writing a shard and reading it back through `make_viewset_dataset` is the
construction rather than a shortcut: it puts the control arm through the same
parse and scalar-normalisation path training used.

**Two things pre-registered now, before the harness exists.**

1. **The offline control arm measures the model minus its DV path.** It injects
   at a synthetic ephemeris with depth 0, so no DV report exists for that
   ephemeris and the `detection` / `ghost` branches run masked
   (`dv_usable=False`). The model handles that by design — it is what the gating
   is for — but it is a real difference from how 56% of training rows were built,
   and it is a property of the measurement, not a defect to discover later.

2. **This does NOT restore comparability with 26.4%, and "must fall" is
   re-specified.** The 26.4% came through the live **dual-view** path: different
   model, different view builder, different preprocessing. An offline
   branch-model control arm is a **new baseline, not a continuation**. For "must
   fall" to be a test rather than a slogan, the **dual-view model is re-measured
   through the same offline harness** and the comparison is made within protocol.
   That is cheap — its checkpoints are on disk.

   This is a criterion being re-specified, and the reason is attached
   deliberately: it is re-specified as **unmeasurable as written**, *before any
   result exists*, rather than adjusted after seeing one. Those are different
   acts and only the second is the thing pre-registration exists to prevent.

**The honest limit, recorded so the number is not over-read later.** This can
never support "the serving defect is fixed" — only "the branch architecture has a
lower control-arm rate than the incumbent **on a common offline protocol**". If
the offline and live protocols disagree, that gap is itself a finding, and it is
only obtainable at stage 11.

**Also settled: the sweep's control is valid.** The re-baseline predates the
branch-drop refactor, so its checkpoints could in principle have been built by
different code. Checked rather than assumed — `branches-20260808-rebaseline`'s
`fold_0/model_0_cnn_branches.keras` against HEAD's no-drop build:

```
params 169,361   layers 134   inputs 13
ordered (layer name, layer class) sequence: IDENTICAL
```

Identical creation order means an identical initialiser RNG draw order, so the
six hours of sweep are not carrying an uncontrolled architecture difference.

*(Stated as the ordered sequence rather than as a digest of it. A digest is only
a fact if the function that produced it is recorded too — two correct checks of
this ran and returned different hashes because they hashed differently, which is
exactly the trap. The comparison above is reproducible from this repo.)*

**Stage 7 *(old D)* — branch attribution.** Which of the twelve branches earn
their place, now that the unfolded one can actually see a transit. A branch-drop
mechanism driven by config rather than by editing `build_cnn_branches` — leaving
a branch out must be a declared experiment, not a code edit — plus the harness
owed since old 2(b) was specified: **injection-recovery on matched hosts with
observation baseline held constant**, the only causal measure available here and
immune to the label-selection confound that moved to stage 8. **Twelve branches
over eleven views group into eight families** — the "eleven branches / ~7
families" this file carried until 2026-08-09 counted views, not branches, and
`BRANCH_FAMILIES` is now derived so the two cannot drift again. One leave-one-out
CV run each at `--n-models-per-fold 3`.
**Its criterion is already specified and is not AUC**: the control-arm host-pass
rate, 26.4%, must fall — **and is currently unmeasurable; see above.**

#### 3.5c Pre-registered before the sweep — recorded 2026-08-09, nothing launched

**Scope: three families, not eight, prioritised by uncertainty rather than
completeness.** A full sweep spends hours confirming that `flux` and `global`
matter, which is not in doubt.

| family | why it is uncertain | params |
|---|---|---:|
| `unfolded` | rebuilt 2026-08-08 (`65d1d98`) and never attributed; the branch stage 7 exists for | −8.5% |
| `periodogram` | built and never measured; largest non-flux capacity delta, and the most plausible redundancy with flux | −14.4% |
| `scalar_only` | tests whether the scalar path earns the `dv_usable` masking complexity | −9.8% |

**This sweep is EXPLORATORY, and that is fixed now rather than after the
numbers.** It reads AUC and recall @1% FPR deltas — which is what stage 4 did
four times without persuading anyone — because the criterion that would settle
the stage is blocked. **No branch is added or removed on the strength of it.** A
family that passes earns a confirmatory repeat, not a decision.

**The threshold is a two-run difference, not a margin against a fixed number.**
Both the control and each ablation are 3-member ensembles, and their errors are
independent, so the spread of the *difference* combines in quadrature:

```
sd(delta) = sqrt( (sd_control/√3)² + (sd_ablation/√3)² )
```

Using the re-baseline's measured floors, and assuming the ablation's own floor is
comparable, the 2σ bars are **recall ≈ 0.048** and **AUC ≈ 0.0098** — *not* the
0.034 and 0.007 that apply to a margin against a fixed reference. Each run
reports its own `pooled_gate_recall_seed_sd` and `seed_sd`, so the bar is
recomputed per comparison rather than assumed.

**Multiple comparisons, stated in advance.** Three families against a 2σ bar is a
familywise false-positive rate of roughly **1 − 0.954³ ≈ 13%** under the null —
and it would have been ~31% at eight. No correction is applied, because the
sweep is exploratory and a Bonferroni bar would make it unable to detect anything
real at n=3; the number is recorded so "one family passed" is read as what it is.

**A null is ambiguous and does not mean "carries no signal".** Leave-one-out
measures whether a branch is *uniquely necessary*. Two branches carrying the same
signal both show ≈0 delta, and the periodogram pair against the flux family is
exactly that shape. **A null reads as "not uniquely necessary" only**; concluding
"this branch is useless" from it would need a joint drop, which is not in this
sweep.

**The capacity confound.** Every drop removes 7–14% of parameters, so a hurt
could be capacity. The capacity arm bounds it — **+19% params moved nothing**
(paired d = −0.44) — which makes capacity a weak explanation for a large hurt but
not a zero one, and it cuts the wrong way for a *null*: a branch that carries
signal could show ≈0 because the capacity it freed was reused.

**Control:** `models/cv/branches-20260808-rebaseline`, identical shards, identical
seed and splits. Runs go to `models/cv/branches-20260809-drop-{family}`. Nothing
promotes; the registry is untouched.

**One arm spanned a hibernation — recorded before either result is read.** The
laptop's battery reached 1% and the machine hibernated from ~09:17 to 11:00 on
2026-08-09 (`pmset`: `Wake from Hibernate ... Using AC (Charge:1%)`; the process
showed 2 h 14 m elapsed against 59 m of CPU). The processes survived and resumed
correctly, and the damage is bounded:

| arm | status |
|---|---|
| `unfolded` | finished 08:48, ~30 min before the hibernation — **clean** |
| `periodogram` | fold 0 clean; **folds 1–4 span the hibernation** |
| `scalar_only` | started after the wake — **clean** |

Hibernation restores process memory exactly, and TF/Metal would be far more
likely to error than to corrupt silently, so this is probably immaterial —
**probably is not a standard this project accepts for a pre-registered
comparison.** `periodogram` is therefore re-run into
`branches-20260809-drop-periodogram-clean`, and **the clean run is the
authoritative one**, fixed here before either number is looked at.

The hibernation-spanning run is **kept, not deleted**, and the two are compared
as a free consistency check. If they differ by more than the floor, that is a
finding about running training across a suspend — worth having, and it is only
available because the first run was not quietly thrown away.

#### 3.5d Sweep result — every arm null, and the one "PASS" is an artefact (2026-08-09)

Four runs against `branches-20260808-rebaseline`, read by the rule fixed before
any of them existed. TESS slice, each run's own `per_mission.TESS` (n=2,399).

| arm | AUC | ΔAUC | bar | | R@1% FPR | ΔR | bar | |
|---|---:|---:|---:|---|---:|---:|---:|---|
| **control** | 0.9202 | | | | 0.2196 | | | |
| `unfolded` | 0.9135 | −0.0067 | 0.0137 | null | 0.2204 | +0.0008 | 0.0423 | null |
| `periodogram` *(hibernation)* | 0.9215 | +0.0013 | 0.0163 | null | 0.2694 | +0.0498 | 0.0509 | null |
| **`periodogram-clean`** | 0.9213 | +0.0011 | 0.0130 | null | 0.2536 | +0.0340 | 0.0339 | *PASS* |
| `scalar_only` | 0.9094 | −0.0108 | 0.0176 | null | 0.2792 | +0.0596 | 0.0768 | null |

**The `periodogram-clean` PASS clears its bar by 0.00008 — 0.23% of the bar — and
should not be treated as a result.** It satisfies the letter of the pre-registered
rule and is reported as such rather than quietly reclassified, but the
classification is driven by the *bar*, not by the effect:

| run | `pooled_gate_recall_seed_sd` | the three draws |
|---|---:|---|
| control | 0.0292 | 0.1857, 0.1842, 0.2355 |
| `unfolded` | 0.0221 | 0.1162, 0.0838, 0.1260 |
| **`periodogram-clean`** | **0.0029** | 0.1623, 0.1630, 0.1577 |
| `scalar_only` | 0.0597 | 0.1494, 0.2551, 0.1540 |

`periodogram-clean`'s three draws happened to land within 0.005 of each other, an
sd **ten times** the control's tightness — and an sd from three draws carries
~40% sampling spread, so this is far more likely three lucky draws than a
genuinely stabler configuration.

**The comparison that settles it:** `scalar_only` has the **largest** recall
delta in the sweep (+0.0596) and reads *null*, while `periodogram-clean` at
+0.0340 reads *PASS* — purely because one run's 3-draw sd came out 20× the
other's. Under this sweep's own machinery, PASS and null are being assigned by
noise in the error bar rather than by effect size. **Treat all four arms as
null**, and note that the pre-registered consequence of a PASS was never a
decision anyway: it earns a confirmatory repeat.

**What the sweep does support.** `unfolded` — the branch stage 7 exists for, and
the one just rebuilt — costs −0.0067 AUC and +0.0008 recall when removed. Per the
pre-registration that is **not uniquely necessary**, *not* "carries no signal";
redundancy with the flux family is the obvious candidate and a leave-one-out
design structurally cannot separate it. A joint drop would, and is not in this
sweep.

**One pattern, named as a hypothesis and nothing more.** Every drop *raised*
shortlist recall while holding or lowering AUC: +0.0008, +0.0340, +0.0596. All
null individually, but three independent arms agreeing in sign is the same shape
as the capacity arm's 0.145 → 0.236 — which stage 6 showed was a real effect on a
retired architecture. Removing capacity appears to trade ranking for shortlist
recall. **Three nulls pointing one way is not a result**, and this exploratory
sweep cannot establish it; it is recorded as a design question for a later
confirmatory run.

**The hibernation left no measurable trace — and this is only knowable because
the affected run was kept.** The two `periodogram` runs differ by **+0.0158** in
recall against a two-run bar of 0.0383, and by **+0.0002** in AUC against 0.0183:
they agree within noise on both. No evidence that suspending training across a
hibernate perturbs it. The clean run remains authoritative as pre-registered.

#### 3.5e Sequencing: should stage 8 come first? — assessed 2026-08-09

**The question, put by Ollie and never assessed.** Stage 8 invalidates stage 7's
attribution numbers — this file says so twice, and it is why stage 8 was moved
ahead of stage 9. So there is a case for doing stage 8 before finishing stage 7,
on the same "do not pay for it twice" argument that rejected pulling stage 11's
serving parity forward.

**Answer: no. Build stage 7's harness now; defer stage 7's remaining
*attribution compute* until after stage 8.** The question conflates two things
stage 7 still owes, and they sit on opposite sides of the argument.

**1. What stage 8 invalidates is numbers, not code — and the numbers at risk are
four nulls.** Stage 7's 15–20 h estimate was dominated by the sweep, and ~6 h of
it has run. Every arm came back null. Deferring the harness to avoid re-measuring
that protects nothing worth protecting.

**2. The harness is stage 8's instrument, not only stage 7's — this is
decisive.** Stage 8's own success criterion is whether baseline dependence falls,
and the clean test of that was pre-registered on 2026-08-06 as pre-commitment (d):
**injection-recovery on matched hosts with observation baseline held constant.**
That is this harness, including the baseline matcher. Doing stage 8 first means
building stage 8's measuring instrument under stage 8's own pressure — the exact
shape of the argument that rejected pulling serving parity forward, running in
the same direction.

**3. Stage 8 needs a before-reading, and this is the only protocol that can
produce one.** The live 26.4% is already recorded as not comparable to anything
the branch line produces. Without the harness in place first, the pre-stage-8
control-arm rate exists only retrospectively. It is *recoverable* — the fold
checkpoints, the bundles and the labels snapshot are all on disk — so this is
insurance rather than a hard block; but it depends on the pre-stage-8 host
population still being reconstructable after `labels.parquet` changes, and on
someone choosing to go back for it once newer numbers exist.

**4. The cost asymmetry is large, and it is measured rather than estimated.** Two
things expected to be missing are already present, and each removes a chunk of
the build:

| needed | status |
|---|---|
| per-fold scalar normalisation constants | **persisted** in `fold_k/cnn_calibrator.joblib` beside the Platt fit |
| out-of-fold routing for a host | **`predictions.parquet` carries `fold`** |

So the remaining build is the driver, the baseline matcher and the report, over a
chain of existing parts, with compute that is forward passes over 6,721 cached
FITS. Against that, stage 8 is 25–35 h at explicitly **low** confidence, gated on
external catalogue ingestion whose only precedent here (the DV download) was 5×
out.

**What is genuinely paid for twice, and the consequence.** The branch-model
control-arm *reading* itself: stage 8 retrains, so the number moves. It is
therefore recorded as a **baseline, not a verdict**, and the consequence is a
scoping recommendation rather than a silent choice — **no further stage-7 CV
compute before stage 8.** No joint drop, no widening to the five unmeasured
families. That work is the part stage 8 really does invalidate, and the previous
session's own "the natural follow-up is a joint drop" should wait behind it.
**Re-scoping is Ollie's call, so this is put as a recommendation and not acted
on.**


### 3.6 Stage 7i — the offline control-arm harness

#### 3.6a Pre-registered before the harness runs — recorded 2026-08-09, nothing built

The two limits already recorded under *Decided 2026-08-09* stand unchanged and
must survive into the result: **`detection`/`ghost` run masked** (`dv_usable`
False — no DV report exists at a synthetic ephemeris), and **this does not
restore comparability with 26.4%**, so the dual-view incumbent is re-measured
through the same harness and the comparison is made within protocol.

Reading the code surfaced three further choices the sweep never had to make.
Fixed here, before the harness exists.

**1. Fold routing is out-of-fold, and a host that cannot be routed is dropped.**
Control-arm hosts are real labelled TESS targets that were in training, so the
only honest protocol is the fold that held each one out — read from the run's own
`predictions.parquet` `fold` column, which carries it. A host absent from that
frame is **dropped, not zero-shot scored**: silently mixing the two protocols
across one population is the exact defect stage 3 closed, and `eval/scoring.py`
already refuses to guess between them.

**2. The pass threshold is the run's own recall @1% FPR operating point, and
both operating points are reported.** A branch run directory stores **no**
threshold — `bundle["threshold"]` is the legacy *serving* bundle's field, and the
branch bundle carries `calibrator`, `platt_a`, `platt_b`, `scalar_constants` and
nothing else. So one must be chosen rather than read:

| candidate | verdict |
|---|---|
| **1% FPR on the run's own out-of-fold TESS rows** | **primary.** It is the threshold every gate decision in this project is made at, and "passes threshold" then means "would reach the shortlist" |
| mean of the folds' F1-optimal thresholds | **reported alongside.** It is what the live path used for the original 26.4%, so omitting it would discard the only continuity available |

Both are applied to the same scores, so reporting the pair costs nothing and
stops the choice of operating point from being an unexamined degree of freedom.
The same rule is applied to the incumbent, computed from its own re-baselined
predictions.

**3. Baseline matching is on `baseline_days`, by quantile stratum, drawn equally
per label.** The statistic of interest is not 26.4% but the **46.7% / 12.3**
split between planet hosts and false-positive hosts — and observation baseline is
the confound that split is most exposed to, at +0.387 against the label on TESS.
So hosts are stratified into `baseline_days` quantile bins and drawn equally from
each label within each bin; strata that cannot supply both labels are dropped and
**the number dropped is reported**, because a matcher that quietly discards its
hard strata reports a clean number about an easier population.

`labels.parquet` has **no `expected_transit_count`** (18 columns, confirmed), so
`baseline_days` — `(expected_transit_count − 1) × period` — is unavailable there
and comes from a join against the viewset scalars.

**The comparison population, measured 2026-08-10 before the run.** Hosts routable
out-of-fold in **both** run directories — an unrestricted pair would compare two
different populations again, which is the defect this project keeps hitting.

| | hosts |
|---|---:|
| cached TESS, depth-filtered | 1,058 |
| routable in `branches-20260808-rebaseline` | 1,057 |
| routable in `ca906040` | 1,052 |
| **routable in both — the comparison population** | **1,051** *(552 FP / 499 planet)* |
| **after baseline matching, 4 strata** | **580** *(290 per label, 0 strata dropped)* |

The per-stratum counts are the +0.387 correlation made concrete, and they are why
matching caps the draw at 580:

| baseline stratum | FP hosts | planet hosts | matched pairs |
|---|---:|---:|---:|
| 0 *(shortest)* | 230 | 33 | 33 |
| 1 | 164 | 99 | 99 |
| 2 | 81 | 181 | 81 |
| 3 *(longest)* | 77 | 186 | 77 |

**580 hosts x 3 periods = 1,740 rows per lane** is therefore the full measurement,
and it is the pool's maximum rather than a chosen sample size.

**Recorded as unmeasured, not assumed away.** Matching costs sample size, and the
cached-FITS host pool is what it is. If a matched draw cannot fill its strata the
result is reported at the size it reaches, with the unmatched rate beside it —
not topped up from unmatched hosts.

#### 3.6b Pre-registered before the measurement — recorded 2026-08-10, nothing run

Both lanes exist and are smoke-tested; **no control-arm number has been produced
for either model.** Written down first because nobody is watching an autonomous
session read its own result.

**What is run.** Two invocations, identical hosts: `--per-stratum 200`,
`--seed 42`, three periods (3, 7, 12 d), each restricted with
`--also-routable-in` to the other run. That yields the same 580-host matched
draw for both — 290 per label, 4 strata, none dropped — so the comparison is
paired on the host.

**The criterion, re-specified as unmeasurable-as-written on 2026-08-09 and
settled here.** "26.4% must fall" cannot be read against the live figure. What
is tested instead:

> **Does the branch model score the *star* less than the dual-view incumbent, on
> one common offline protocol?**

**The bar, computed before the numbers exist.** A pass rate is a proportion over
hosts, and the three periods of one host are correlated, so the effective n is
the **host** count and not the 1,740 rows. At n=580 a single rate carries
`2σ ≈ 0.04`; at n=290 per label, `2σ ≈ 0.06`; the planet-minus-FP split combines
in quadrature to `2σ ≈ 0.07`. The comparison between models is **paired on the
same hosts**, which makes an unpaired bar conservative — it is used anyway, and
the pairing is noted rather than banked.

**How each outcome reads — fixed now.**

| outcome | reading |
|---|---|
| branch rate **below** incumbent by more than 0.04 | the branch architecture scores the star less, **on this protocol only**. It cannot support a serving claim, and it is not a promotion argument — the gate is AUC and shortlist recall |
| within **±0.04** | **level.** The branch architecture does not reduce host-scoring. Given stage 4 rejected every arm on recall, this is the outcome that closes the branch line's remaining case |
| branch rate **above** incumbent by more than 0.04 | the branch architecture scores the star *more*. Report it; do not re-specify |

**The split is the sharper statistic and is read alongside.** The headline 26.4%
conflates two populations; the 46.7 / 12.3 gap is what "the model scores the
star" actually predicts. A model that vets the transit should show a **smaller
planet-minus-FP split**. Because hosts are baseline-matched, a residual split
can no longer be explained by observation baseline — which is the whole reason
the matcher exists.

**Predictions, recorded so they can be wrong.**

1. Both models pass **well under 26.4%** at the shortlist cut, because that cut
   is far stricter (1% FPR, ~0.96–0.97 calibrated) than the F1-optimal cut the
   26.4% was measured at. The F1 cut is the one to compare against 26.4% loosely.
2. Both show a **positive** planet-minus-FP split at the F1 cut — the pathology
   is real and matching removes the baseline confound, not the effect.
3. The two models land **within 0.04 of each other** on the overall rate. Stage
   4 found the branch line level on TESS AUC and worse on recall; there is no
   measured reason to expect a large control-arm separation.

**Nothing promotes.** `models/registry.json` untouched, `ca906040` stays served.
A favourable number here does not reopen stage 4.

#### 3.6c Result — the branch architecture does not score the star less (2026-08-12)

`results/control_arm/`. **580 baseline-matched hosts x 3 periods = 1,740 rows per
lane, 0 unscored**, exactly the sizing pre-registered before the run. Both limits
recorded on 2026-08-09 survive into the result and are written into the output
JSON: `detection`/`ghost` ran masked, and these numbers cannot support a claim
about *serving*.

**Pass rates, with the host as the inferential unit.** The rates below are the
driver's row-level output over 1,740 rows; the bars beneath them use the **host**
count of 580, because a host's three periods are correlated and the row count
would overstate the precision. *(Corrected 2026-08-12: an earlier version of this
line said a host's three periods are averaged before the population mean. They
are not — no averaging happens anywhere in the driver. Row-level and host-level
agree to every digit here only because every host contributed exactly three
scored rows and none were dropped; they diverge as soon as
`n_unscored_dropped` is non-zero, which the driver permits. Recorded rather than
left to hold by luck.)*

| | cut | pass | planet hosts | FP hosts | **split** |
|---|---:|---:|---:|---:|---:|
| incumbent, shortlist | 0.9623 | **0.0000** | 0.0000 | 0.0000 | +0.0000 |
| branch, shortlist | 0.9731 | **0.0006** | 0.0000 | 0.0011 | −0.0011 |
| incumbent, F1 | 0.5390 | 0.1230 | 0.1839 | 0.0621 | **+0.1218** |
| branch, F1 | 0.4486 | **0.1943** | 0.2540 | 0.1345 | **+0.1195** |

2σ bars: **0.036** on a single rate (n=580), **0.051** per label (n=290),
**0.072** on the split. Paired on the host, `2×se = 0.0344`.

**The pre-registered PRIMARY operating point returned a floor, and that is a
finding about the protocol rather than a result.** At the 1% FPR cut *both*
models pass essentially nothing — 0.0000 and 0.0006. There is no room for a rate
to fall, so the comparison that was nominated as primary **cannot carry the
stage's criterion**. It is reported as level because it is, and because the
alternative — quietly promoting the secondary point to primary after seeing the
numbers — is the thing pre-registration exists to prevent. The F1 point was
pre-registered as *reported alongside*, so it is available without
re-specification.

**At each model's own operating point the branch model scores the star MORE.**
Paired over the same 580 hosts, `+0.0713` against a paired bar of `0.0344` —
**2.1x the bar**. Per the pre-registered outcome table this reads as *"the branch
architecture scores the star more. Report it; do not re-specify."*

**The split — the sharper statistic — is level.** +0.1195 against +0.1218, a
difference of **−0.0023 against a 0.072 bar**, which is 3% of it. The 46.7 / 12.3
pathology is reproduced in both models at essentially identical magnitude, on
hosts where observation baseline has been matched away. Whatever drives
host-scoring, **eleven diagnostic branches do not reduce it.**

**Three predictions, two right and one falsified.**

| # | prediction | outcome |
|---|---|---|
| 1 | both pass well under 26.4% at the shortlist cut | **correct**, and by more than expected — both ≈0 |
| 2 | both show a positive planet-minus-FP split at F1 | **correct** — +0.1218 and +0.1195 |
| 3 | the two models land within 0.04 on the overall rate | **FALSIFIED at the F1 point** (+0.0713). Correct at the shortlist point, where the floor makes it trivial |

**A diagnostic, explicitly NOT pre-registered and not part of the reading.**
Scored at the *incumbent's* cut rather than its own, the branch model passes
0.1075 against 0.1230 (paired −0.0155, inside the bar) with a split of +0.0747.
So much of the +0.0713 is where each model's F1 optimum sits — the branch
model's is lower (0.4486 vs 0.5390) — rather than host-scoring alone. This is
post-hoc, it does **not** overturn the pre-registered reading, and it is recorded
because omitting it would overstate the result.

**Verdict: stage 7's criterion is NOT met.** "The 26.4% control-arm host-pass
rate must fall", re-specified on 2026-08-09 as a within-protocol comparison, is
answered **no**: the split is level and the own-operating-point rate is higher.
Combined with five arms rejected on shortlist recall, **the branch line now has
no measured advantage on either of its two criteria.** That is the outcome the
pre-registration named as closing its remaining case.

**Two defects were found and fixed while running this, and the numbers above are
post-fix.** Both are the house failure mode — a plausible number from a broken
computation:

1. **Score collapse across periods.** `score_through_run` assigned by `tic_id`
   match, but `write_viewset_shards` permutes rows, so positional alignment was
   unavailable and every row of a host matched. All three of a host's periods
   were overwritten with whichever was processed last — two thirds of the
   measurement silently discarded, with a wholly plausible pass rate surviving.
   Fixed by aligning against the index the writer actually produced, with a
   raise if the stream and index disagree. The incumbent lane never had this bug,
   which is how it surfaced: its per-host scores differed and the branch lane's
   did not.
2. **The stream was iterated four times per fold** — once per member plus once
   to read identities back. Folded into the members' own passes. Scoring went
   from **~10 minutes per fold to ~14 seconds**.

**Cost, for future sizing.** Build ~50 min for 1,740 rows (dominated by two BLS
periodograms per row; the multi-sector tail is much slower than the median),
scoring ~1 min. The incumbent lane is ~10 min end to end. `--shard-dir` keeps the
built shard set on disk instead of a temp dir, so a failed scoring pass leaves
something to inspect and re-score by hand. *(Corrected 2026-08-12: it does **not**
skip the build on a re-run — an earlier version of this line said it did. The
driver rebuilds unconditionally, on purpose: a directory left by a different host
draw, seed or period list would otherwise be scored as though it were this one,
which is a wrong measurement wearing a plausible pass rate. Thirty-five minutes
is the cheaper side of that trade.)*

**Nothing promotes.** `models/registry.json` untouched; `ca906040` stays served.

### 3.7 The promotion gate was not calibrated to its own noise floor (2026-08-12)

Audited on Ollie's instruction — *"inspect the promotion and rejection gates as
well, I don't want any bias"* — rather than found by a failing test. **Three
defects. None changes a recorded verdict; all three changed what a verdict was
allowed to claim.**

**1. The gate could not promote anything, and had not been able to since the
re-baseline.** `registry.json` names `ca906040`'s own `cv_summary.json`, which
carries no `per_mission` block. So `_gate_slice` returns `None`,
`_population_mismatch` fires, and `evaluate_promotion` returns REJECT **before a
single metric is compared**. Run, it says so in as many words:

> REJECT: gated on pooled CV means — a summary here predates the per_mission
> block; … refusing to promote on a pooled comparison whose rows cannot be
> matched

The 2026-08-07 audit predicted exactly this and noted `promotion_gate.py` had no
flag to point at a re-baselined summary. Stage 3 built the summary and the note
was never actioned, while the stage-3 row above recorded the opposite. **Every
branch rejection on this project was therefore decided by hand, via
`evaluate.py compare`, not by the gate.** Closed with `--incumbent-summary`. The
registry is deliberately left alone: it names what is *served*, and making the
serving pointer depend on an evaluation artefact is the wrong coupling — besides
which touching it is a stop-and-ask.

**2. The recall tolerance was tighter than the floor this project had already
measured.** Stage 6 measured `pooled_gate_recall_seed_sd = 0.0292`, fixed the
rule `2 × seed_sd / √n_models`, and recorded the conclusion in this file: *a
recall @1% FPR margin under ~0.034 is not a decision.* The gate went on
rejecting at a hardcoded **0.02** — 0.68 σ — so a candidate whose true recall
equalled the incumbent's was failed by reseeding noise about **31%** of the time,
on the single criterion that has rejected every branch arm. The constant predates
stage 6 and was never revisited when it landed. `recall_tolerance` now defaults
to the candidate's own measured floor.

**3. A tie on AUC was recorded as a defeat.** AUC was a bare `>` with **zero**
tolerance while Brier (+0.005), ECE (+0.01) and recall (−0.02) each granted a
band. Whoever held the incumbent seat therefore won every tie by construction —
at a run-level AUC floor of 0.0069, a genuinely level candidate lost a coin flip.
The *verdict* is unchanged and should be: a tie is not a reason to churn a
deployed model. What changed is the record — a delta inside the floor now reads
*"level on ROC-AUC … not a measured difference"* instead of *"does not beat the
incumbent's CV score"*, and every margin is quoted as a multiple of its floor.

**The symmetry test, run because a gate that only ever rejects challengers is
indistinguishable from one that is rigged.** Same two summaries, roles swapped:

| direction | verdict | rejected on |
|---|---|---|
| branch as candidate | REJECT | shortlist recall 0.220 vs 0.307 |
| **incumbent as candidate** | **REJECT** | **AUC 0.9100 vs 0.9202** |

**The gate is not protecting `ca906040`.** On the re-baselined like-for-like
comparison the branch model *wins* TESS AUC (+0.0102), Brier (−0.0103) and ECE
(−0.0279), and loses only on shortlist recall. The bias was in the calibration of
the thresholds, not in their direction.

**Re-run after the fix, the rejection stands and is now measured:**

> recall @1% FPR 0.220 vs incumbent 0.307 (−0.0873, **2.6× the 0.0337 floor**;
> tolerance 0.0337) — shortlist recall degraded beyond tolerance

`paired_folds` remains computed-but-never-gating, and returns `None` against the
re-baselined summary in any case because that summary carries no `folds` block.
**Recorded as a known limitation, not fixed** — making the project's best
statistic decisive is a change to what the gate *means*, and that is Ollie's
call, not a bug fix.

**Nothing promotes.** `models/registry.json` untouched; `ca906040` stays served.

### 3.8 Observation baseline — a real problem architecture cannot fix

Measured 2026-08-05, baseline as a span in **days**:

| population | corr(score, baseline) |
|---|---:|
| incumbent, 3,908 scored candidates | **+0.208** (+0.187 controlling period) |
| incumbent, labelled CV set | +0.238 |
| stage 4 branches, labelled CV set | +0.239 |
| **the ground-truth label itself** | **+0.278**, and **+0.387** on TESS alone |

The correlation survives inside every TESS period band and is not a period
artefact. TESS confirmed planets have a median baseline of **1,495 d against
430 d** for false positives.

**"Every model sits below the labels" was true on 2026-08-05 and is not true
now.** This line said so until 2026-08-12 and had to be corrected twice over,
because the crossing had already been recorded elsewhere in this file and never
propagated back here. Stage 6 noted the re-baseline reached **+0.3025 pooled,
above the +0.278 label figure**. Re-measured on 2026-08-12 with the same Spearman
statistic, **per mission**, which is what pooling was hiding:

| series | all missions | **TESS** *(gates)* | Kepler | K2 |
|---|---:|---:|---:|---:|
| branch model, `branches-20260808-rebaseline` | +0.3025 | **+0.5155** | +0.0859 | −0.0064 |
| incumbent `ca906040`, shared TESS rows | — | +0.3812 | — | — |
| **the ground-truth label**, same rows | +0.2136 | **+0.3874** | +0.1025 | −0.1490 |

The label's TESS figure reproduces the recorded +0.387 to three places, so the
slice is right and it is the *model* row that was stale. **On the mission that
gates, the branch architecture sits +0.13 above the labels it learned from** —
it does not merely inherit the confound, it amplifies it. The incumbent, at
+0.3812, still sits just below.

**Consequence for stage 8: there are two targets, not one.** The bias in the
labels, and the branch architecture's amplification of it. An intervention that
fixes the first and leaves the second is a partial result, and the pre-registration
must be able to tell them apart.

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


### 3.9 Stage 8 — labels and negatives

**Why this stage exists, and why it was moved.** Recorded 2026-08-08,
before any of it ran.

**Stage 8 *(old 3)* — labels and negatives.** EB-catalogue and brown-dwarf
negatives, the ephemeris-match test, and scrambled/inverted synthetic negatives
built with our existing injection machinery. Plus the observation-selection
problem below, which arrived here from the branch-model work.

**Moved ahead of stage 9 on 2026-08-08.** Two reasons. Stage 8's
interventions change the training distribution, so anything measured before it
has to be re-measured after — and stage 9 is the expensive one, so the roadmap
order paid for it twice. And on the evidence, two architecture runs have now
been rejected while the largest measured defect sits at **+0.278 in the labels
themselves, +0.387 on TESS** — above every model, and somewhere no architecture
can reach. The original ordering was set before either of those facts existed.

Two knock-ons to expect: changing the label distribution invalidates the
re-baselined incumbent summary from stage 3, which will need regenerating — one
command, and a reason to keep stage 3 a repeatable path rather than a one-off
artefact — and it invalidates stage 7's attribution numbers, which is exactly why
stage 8 sits ahead of stage 9.



#### 3.9a Pre-registered before stage 8 runs — recorded 2026-08-12, nothing built

Written before any intervention exists, because nobody is watching an autonomous
session read its own result.

**The before-readings already exist and are NOT to be re-derived.** Stage 7i
produced the control-arm numbers; the baseline sensitivities were re-measured
per mission on 2026-08-12. Everything below is compared against these:

| statistic, TESS gate slice | before |
|---|---:|
| Spearman(branch score, `baseline_days`) | **+0.5155** |
| Spearman(label, `baseline_days`) | **+0.3874** |
| **amplification gap** = score − label | **+0.1281** |
| control-arm split, branch, F1 cut | **+0.1195** |
| TESS AUC / recall @1% FPR | 0.9202 / 0.2196 |

**Two targets, separated. This is the point of the decomposition.** Stage 8's
framing until now was "the bias is in the labels, no architecture can reach it".
That is half the story:

- **Target A — the labels.** `Spearman(label, baseline_days)`. The interventions
  change what the training labels *are*, so this moves by construction and its
  movement is the intervention working as designed.
- **Target B — the amplification.** `score − label`. The branch model sits
  **+0.1281 above** its own labels on TESS. No label intervention is required to
  move this, and an intervention that fixes A while leaving B is a partial
  result. Reporting only the score correlation would conflate the two, which is
  how the pooled figure hid the crossing for four days.

**The evaluation population is frozen, and this is the trap to avoid.** Adding
negatives changes the training population, so a correlation computed over the
*augmented* set is not comparable to +0.5155. Every number above and after is
measured on the **same out-of-fold TESS rows** as the before-reading. Synthetic
and external negatives enter **training only** and are excluded from every
evaluation slice. A run that cannot demonstrate that exclusion is not read.

**The bar.** Sampling error on a Spearman ρ at n=2,399 is `1/√(n−3) ≈ 0.0204`,
so **2σ ≈ 0.041** — but that is a fixed model's sampling error, and reseeding
noise on this statistic has never been measured here. So each run reports the
**per-member spread** of its own baseline sensitivity at `--n-models-per-fold 3`,
and the bar is stage 6's rule — `2 × sd / √3` — floored at the 0.041 sampling
bar. **A margin under the larger of the two is not a decision.**

**Arms, and why they are separate runs.** Three interventions run together
cannot be attributed, and this project has already paid for that once with the
three-family sweep. Each is measured against the same control:

| arm | what changes |
|---|---|
| control | the current distribution, re-run for a same-code comparison |
| **P** propensity weighting | per-example weights ∝ inverse propensity on `baseline_days` |
| **S** stratified negatives | negatives resampled to match the positives' baseline distribution |
| **N** synthetic negatives | scrambled + inverted light curves, baseline-independent by construction |
| combined | only if at least one single arm clears its bar |

At ~70 min a run that is ~4.7 h for the four, inside the 6–8 h budgeted.

**How each outcome reads — fixed now.**

| outcome | reading |
|---|---|
| amplification gap **falls** beyond the bar | the architecture's contribution to baseline dependence is reachable. Report the arm and the residual |
| gap **level**, label correlation falls beyond the bar | the label intervention worked and the architecture still amplifies. **A partial result, reported as partial** — this is the outcome the old framing would have called a success |
| **neither** moves beyond its bar | the intervention is falsified. Record it; do not re-specify, and do not run a fourth arm looking for a better one |
| baseline dependence falls but **TESS recall @1% FPR falls beyond its 0.0337 floor** | the fix costs shortlist performance. Report both numbers together; a bias fix that defeats the deployment use is not a fix |

**Predictions, recorded so they can be wrong.**

1. **N (synthetic) moves the amplification gap the most**, because it is the only
   arm that breaks the correlation by construction rather than reweighting an
   existing population.
2. **P (propensity weighting) moves the label correlation but not the gap** —
   reweighting changes what the model is fitted to, not how it extrapolates.
3. **At least one arm costs shortlist recall beyond its floor.** Observation
   baseline is genuinely predictive of the label in this catalogue; removing the
   model's access to it should cost measured performance, and an intervention
   that costs nothing would be evidence it did nothing.
4. The control arm's split (+0.1195) **does not move** on any arm, because
   host-scoring and baseline dependence are different defects — stage 7i already
   showed the split survives baseline matching.

**The kill criterion stands as a decision already made.** If external catalogue
ingestion exceeds **~8 hours** without a usable negative set, fall back to the
synthetic negatives alone. Not re-litigated here — executed.

**Nothing promotes.** Whatever stage 8 measures, `models/registry.json` is
untouched and `ca906040` stays served.

#### 3.9b Result — the amplification is reachable, the labels are not (2026-08-13)

Four arms, `--n-models-per-fold 3`, ~2 h each. **Prediction 4 was measured on
2026-08-14** through the stage 7i harness and is written up below; the stage is
now complete.

**The evaluation population is the full out-of-fold TESS slice the before-reading
was taken on, n=2,399, identical rows for every arm.** Recorded because the first
attempt at this table intersected the four arms' rows instead, and that is wrong
in a way worth naming: **the intersection *is* the stratified arm's kept rows**,
a population that arm engineered to be free of the confound. Its own label
correlation there is **+0.0573 against the slice's +0.3874**, so comparing arms on
it would have refereed the contest with one contestant's own instrument.

| arm | TESS AUC | recall @1% FPR | score↔baseline | label↔baseline | **gap** |
|---|---:|---:|---:|---:|---:|
| control | 0.9204 | 0.2506 | +0.5139 | +0.3874 | **+0.1265** |
| **P propensity** | 0.9138 | 0.2642 | **+0.3803** | +0.3874 | **−0.0071** |
| N synthetic | 0.9127 | 0.2460 | +0.5097 | +0.3874 | +0.1223 |

**The control reproduces the before-reading**: score +0.5139 against +0.5155, gap
+0.1265 against +0.1281. An independent retrain landing on the same numbers is
what makes the rest of the table readable.

| arm | Δ gap | vs bar | Δ AUC | vs floor | Δ recall | vs floor |
|---|---:|---|---:|---|---:|---|
| **propensity** | **−0.1336** | **3.3×** | −0.0066 | 0.8× *(level)* | +0.0136 | 0.3× *(level)* |
| synthetic | −0.0042 | 0.1× *(null)* | −0.0076 | 0.8× *(level)* | −0.0045 | 0.1× *(null)* |

Bars are each run's own per-member spread by stage 6's rule, floored at the
0.0409 Fisher sampling bar, exactly as pre-registered.

**Propensity weighting eliminated the architecture's amplification of the
confound at no measurable cost.** The branch model went from sitting **+0.13
above** its own labels to fractionally below them: it no longer amplifies, it
merely inherits. That is the pre-registered outcome *"the architecture's
contribution to baseline dependence is reachable"*.

**What it does not claim.** Target A — the bias in the labels — is untouched, and
cannot be otherwise: the label correlation on a frozen evaluation slice is
+0.3874 by definition. What is gone is target B, the amplification. Stage 8's
deliverable is therefore **half of what the stage set out to reach, and it is the
half no architecture change could have delivered.**

**Arm S is not comparable, and that is structural rather than a failure.** It
never scored 680 of the 2,399 pre-registered rows, because rows dropped before
the split are absent from training, validation *and* test. Its own-slice figures
(AUC 0.8799, recall 0.0777) are measured on a rebalanced population where the 1%
FPR threshold means something else, and they are not evidence about the
intervention. The build-time note that excluding rows from test was "the honest
reading" was correct and incomplete: it also makes the arm unreadable against a
fixed evaluation slice. **A resampling intervention has to keep the evaluation
population whole even when it changes the training one.**

**All four predictions falsified.**

| # | prediction | outcome |
|---|---|---|
| 1 | N moves the gap most | **falsified** — N moved it least (0.1×), P most (3.3×) |
| 2 | P moves the label correlation but not the gap | **falsified**, exactly backwards |
| 3 | at least one arm costs shortlist recall beyond its floor | **falsified** — neither comparable arm did |
| 4 | the control-arm split does not move | **falsified** — it fell −0.0966, 1.3× its bar. See below, and read the limit with it |

**Prediction 3 deserves its explanation, because it was written as a trap and the
trap caught the wrong thing.** The pre-registration reasoned that an intervention
costing nothing is evidence it did nothing. P plainly did something — 3.3× its
bar — and cost nothing. The prediction conflated two quantities: removing
**label-level** dependence must cost performance, since baseline genuinely
predicts the label in a test set drawn from those labels; but removing only the
**amplification** — the part where the model used baseline *more than the labels
justify* — is free by construction, because over-use beyond the labels carries no
predictive power on a test set built from them. Costless was the correct
expectation for what P actually did.

#### 3.9c Prediction 4 — the split fell, and the construct behind it did not (2026-08-14)

The stage 7i harness on both `stage8-control` and `stage8-propensity`: **580
baseline-matched hosts x 3 periods = 1,740 rows per lane, 0 unscored**, the same
sizing as stage 7i. The draw is byte-identical to stage 7i's — both arms' fold
maps match `branches-20260808-rebaseline`'s exactly, so the seed-42 matcher
reproduces the same host set. Verified by `tic_id` checksum in pandas *before*
launching, rather than read out of the log afterwards.

| lane | F1 cut | pass | planet hosts | FP hosts | **split** |
|---|---:|---:|---:|---:|---:|
| stage 7i, `branches-20260808-rebaseline` | 0.4486 | 0.1943 | 0.2540 | 0.1345 | **+0.1195** |
| stage 8 control | 0.4047 | 0.2730 | 0.3575 | 0.1885 | **+0.1690** |
| stage 8 propensity | 0.3841 | 0.2201 | 0.2563 | 0.1839 | **+0.0724** |

**Prediction 4 is falsified.** Propensity against its own same-code control is
**−0.0966**: **1.3x** the 0.0720 split bar pre-registered in stage 7i, and
**1.6x** a 0.0592 paired bar computed on these hosts. A paired bootstrap over the
580 hosts gives 95% CI **[−0.157, −0.039]**, not crossing zero. Per the
pre-registered outcome table this reads as *"propensity weighting reduced
host-scoring as well as amplification. A second, independent win."*

**Running the control was load-bearing.** Stage 8's control splits at **+0.1690**,
not the +0.1195 stage 7i measured on the same hosts with the same code — a
**+0.0494** move from reseeding alone. Against the historical +0.1195 the
propensity arm's +0.0724 is −0.0471, *inside* the bar and readable as level. The
pre-registered comparison is against the same-code control, which is precisely
what the control arm exists for; had only the propensity arm been run, that drift
would have been credited to the intervention.

**The limit, recorded because omitting it would overstate the result.** The split
is a *thresholded* statistic and the arms do not sit at the same operating point.
Threshold-free, on the same hosts, the model's ability to tell a planet host from
an FP host on a transit-free light curve is **unchanged**:

| lane | host-AUC, transit-free | 95% CI |
|---|---:|---|
| stage 7i rebaseline | 0.5876 | 0.5429–0.6329 |
| stage 8 control | 0.6234 | 0.5812–0.6688 |
| stage 8 propensity | 0.6045 | 0.5566–0.6451 |

Paired over the 580 hosts, propensity minus control is **−0.0190, 95% CI
[−0.060, +0.019]**, crossing zero at p≈0.33 — while the split difference over the
*same* resamples does not. Propensity's scores on this population are shifted
down (median 0.187 against 0.248) and its F1 cut sits at the **78.0th** percentile
of them against the control's **72.7th**; a stricter operating point mechanically
shrinks a planet-minus-FP pass split.

**So the pre-registered statistic moved and the construct it stands for did
not.** This is post-hoc and does **not** overturn the pre-registered reading —
the same discipline stage 7i applied to its own common-cut diagnostic. What it
does is set the terms on which the win may be banked: **not until a
threshold-free measurement confirms it.** Recorded as a **qualified** second win,
and the qualification is not optional.

**The mechanism prediction 4 assumed does not hold either, and the reason is
worth keeping.** It reasoned that baseline dependence and host-scoring are
different defects because stage 7i showed the split survives baseline matching.
But matching removes the confound from the *host draw*; it does not remove the
*model's* learned reliance on baseline, and those are different operations. On
these matched hosts the control's score↔baseline correlation is **+0.0269** — the
matcher has already closed that channel (residual corr(baseline, label)
**+0.0452**), so there was nothing left there for the intervention to remove.
Propensity's is **−0.2528** on the same inputs while remaining **+0.3803** on the
evaluation slice: the arm's baseline response **changes sign between the two
populations**. Because the matcher's residual leaves planet hosts at a slightly
longer median baseline (1,259 d against 1,118 d), a negative dependence suppresses
planet hosts preferentially — consistent with the observed asymmetry, planet-host
pass **−0.1011** against FP-host pass **−0.0046**. These are transit-free
synthetic inputs and out of distribution, so the sign flip is not by itself a
defect; it is unexplained, and it is the most likely thing driving the split.

**One number here rests on a single draw.** The split's reseeding noise is
characterised by exactly one control retrain (+0.0494), and that is half the
effect being claimed — the same thinness already recorded against the three-draw
floors below.

Reproduction: `~/Downloads/.stage8-scratch/prediction4.py` rebuilds **the split
table and its bars** from the two `results/control_arm/stage8-*.parquet` files.
**It does not rebuild the threshold-free block** — the host-AUCs, either
bootstrap CI, the median/percentile diagnostics or the sign-flip correlations
have no recipe. *(Corrected 2026-08-15: this line previously claimed "every
figure in this subsection", which was false. The audit re-derived the
threshold-free block independently and it is correct — 0.5876 / 0.6234 / 0.6045,
paired −0.0190 with a CI crossing zero — but nothing in the repo re-derives it.)*

**Two defects in the run's own record, found while reading it.**

1. **`run_config` does not record the shard directory.** Arm N differs from the
   control only in which shard set it read, and the summary cannot say so — the
   two are distinguishable solely by `n_examples` (5,767 vs 5,426). A provenance
   gap: fix before any future arm selects its data by path.
2. **The measured floors came in roughly double stage 6's** — the control's
   pooled gate-recall floor is **0.0720** against stage 6's 0.0337. Three draws
   is a thin sd and it is being asked to carry decisions; worth widening before
   the next stage leans on it.

**Cost, for future sizing.** ~2 h per arm at `--n-models-per-fold 3` on this
machine, against the ~70 min the 2026-08-08 handover quoted — that figure is out
by ~1.7× and should not be used for planning. The prediction-4 harness came in
**at** its estimate for once: 48 min for the control arm and 49 for propensity,
1 h 37 for both, against the ~50 min per arm stage 7i recorded.

**The stage-3 incumbent summary needed no regeneration, and the reason it was
thought to is worth recording.** It was carried as outstanding on the grounds
that *"the label change invalidates it"*. Re-derived 2026-08-14: the summary
regenerates **byte-identical** to the committed
`models/cv/incumbent-rebaselined/cv_summary.json`, because the 2026-08-08 label
refresh flipped **zero** labels across the 5,703 `tic_id`s common to
`labels.previous.parquet` and `labels.parquet` — it added two rows and changed
nothing else, and neither added row is in the incumbent's scored set.

> **This no longer reproduces against the working tree, and the reason is not a
> defect in the claim.** Audited 2026-08-15: a catalogue refresh ran at 09:00 on
> 2026-08-15 and rotated `labels.parquet` into `labels.previous.parquet`, so the
> working tree now shows **5,705 common / 0 added**, which reads as a
> contradiction. Against the DVC pointer that was current when the claim was
> written the figures are **exactly** 5,703 common / 0 flipped / 2 added. The
> claim is correct; the artefact under it moved. See 5.4. **And had
labels moved, `summarise` could not have propagated it**: `load_predictions`
re-joins only `mission`, taking `y_true` from the predictions parquet, so the
label change would have needed `evaluate.py score`, not `summarise`. The
anticipated label change was group (a), which was never run. The deliverable
stands satisfied; the rationale attached to it did not survive checking.

**What stage 8 deliberately did not do — decided by Ollie, 2026-08-14.**

1. **External catalogue negatives (group a) were never started and will not be.**
   EB and brown-dwarf catalogues plus the ephemeris-match test, budgeted at up to
   8 h against an 8 h kill criterion. Arm P had already delivered the stage's
   reachable deliverable, so the marginal case was weak. **Recorded as
   deliberately not done, not as an oversight** — the negatives it would have
   produced remain available to any later stage that wants them.
2. **Arm S is not re-run.** Restoring the 680 dropped rows to the *test* split
   would make it comparable for ~2 h of compute, and the question it would answer
   has already been answered by P. The structural lesson stands in its place: **a
   resampling intervention has to keep the evaluation population whole even when
   it changes the training one.**

**Nothing promotes.** `models/registry.json` untouched; `ca906040` stays served.


### 3.10 The UI scaffold, audited against the API — 2026-08-13


A design scaffold exists (manus-generated, five pages: Home, Catalogue, Vetting,
Model Performance, Upload). It is a **scaffold, not the product** — every number
in it is a placeholder. Audited here for one reason only: the plan states the
sequencing test for stage 12 as *"if this step needs a number the API cannot
produce, a step before it was not finished"*, and running that test now is cheap
while running it after stage 11 is not.

**The test fires in both directions, and the reverse one is the finding.**

**1. The design has nowhere to put stage 11's headline deliverable.** Vetting has
three tabs — light curve, probability history, diagnostic flags — and **none of
them is "why did the model score this".** Branch-occlusion contributions are
10–15 h of stage 11 and the justification for the whole stage ("ExoMiner's
explainability story, made interactive"). **Resolved 2026-08-13: the design takes
a fourth tab for per-branch evidence.** Recorded because a stage whose output has
no consumer is a stage that will be built and then not shipped.

**2. The probability-history panel is not a scientific quantity.** It plots one
candidate's P(planet) against *training epoch*. A candidate's score at epoch 7 has
no interpretation, and it is not stored. The **intent** — show the uncertainty —
is right, and two better versions already exist in `ScoreResponse`: `per_fold`
(five fold models' disagreement about this target, which is real epistemic
uncertainty) and `prob_std` (MC-dropout spread). Same visual, real meaning, data
already there.

**3. Per-example uncertainty is available live and not persisted.** `prob_std` is
in `ScoreResponse`; it is **not** in the catalogue. That is W6, and the design
displaying "±0.028 CI" promotes it from a finishing touch to a stage-11
requirement. Cheapest to do while the training path is already open for stage 8.

**4. Training curves are not persisted, and should be.** `cv_summary.json` holds
per-fold metrics, not per-epoch loss/accuracy. The design wants them and they are
independently worth having — early stopping cannot be diagnosed from a summary
that never records where it stopped. Cost is trivial (5 folds x 3 members x ~40
epochs x 4 series ≈ 2,400 floats). **Not implemented on 2026-08-13 because the
stage-8 arms were mid-flight**: the runner launches a fresh interpreter per arm,
so editing the trainer would have made the control arm and the intervention arms
run different code — the exact comparability defect this project keeps paying
for. Queued for immediately after the block.

#### 3.10a Which mission the console reports — decided 2026-08-13

The scaffold shows **ROC-AUC 0.955** as its headline on two pages. That is the
pooled all-mission figure, and it should not be the headline. **The objection is
not that it includes Kepler and K2.**

**The pooled number has arbitrary weights.** Kepler enters the training set at
exactly 1,250/1,250 *by construction* — a sampling decision, not a measurement of
anything. Change that draw and the "headline AUC" moves without the model
changing at all. A weighted average whose weights are a choice is not a property
of the model, which is why `promotion_gate.py` reports `AGGREGATE_SLICE` and
explicitly never gates on it.

**Kepler and K2 are training data; TESS is the serving population.** Kepler
(2009–2013) and K2 (2014–2018) are finished missions whose candidates are already
vetted — nobody will ever ask this system to score a *new* Kepler target. They are
in training because they are the best transit photometry ever collected and they
teach the model what a transit looks like. TESS is ongoing, and every candidate
the service will ever score is a TESS candidate. That is why TESS is
`GATE_MISSION` and the other two are `DIAGNOSTIC_MISSIONS` — reported on every
run, alarmed on a drop beyond 0.02, never blocking.

**Gating on the pooled figure would be actively dangerous**, and not
hypothetically: it is the shape of stage 4. A model can improve on Kepler, which
carries no serving consequence, while degrading on TESS, which carries all of it,
and the pooled mean can rise through both.

**So the console shows all three, separately, with the gating one identified** —
not one blended number. For the served model `ca906040`:

| slice | n | ROC-AUC | recall @1% FPR | role |
|---|---:|---:|---:|---|
| **TESS** | 2,367 | **0.9100** | **0.3069** | **gates — the deployment population** |
| Kepler | 2,238 | 0.9914 | 0.7986 | diagnostic |
| K2 | — | — | — | **does not exist out-of-fold** |
| *pooled* | *4,605* | *0.9558* | *0.4450* | *reported, never gates* |

**The K2 row is the reason this matters in practice.** `ca906040` predates K2
entirely: its only K2 numbers are **zero-shot**, a different protocol on a
population it never trained on. A console that prints a K2 AUC for the served
model would be quoting a zero-shot figure beside two out-of-fold ones and
inviting the reader to compare them. Zero-shot slices are labelled as such or
they are not shown.

Branch models *do* carry an out-of-fold K2 slice — `branches-20260808-rebaseline`
reads TESS 0.9202 / Kepler 0.9547 / K2 0.9351 — so the console's mission panel has
to be driven by whichever model is served rather than by a fixed three-column
layout.

#### 3.10b Two things the project already computes and the scaffold does not show

**Follow-up prioritisation.** `CandidateRow` already carries `tsm`, `esm`,
`teq_k`, `insolation_earth` and the habitable-zone edges — Kempton (2018)
spectroscopy metrics, i.e. *is this worth JWST time*. The product's stated purpose
is ranking candidates for follow-up, and to an observer those are more
decision-relevant than P(planet) alone.

**The observation-baseline caveat.** Stage 8 addresses the largest measured defect
in the project and the console represents it nowhere. A per-candidate indicator
that its score may be inflated by observation baseline would make the stage
visible to a user instead of buried in this file.

#### 3.10c Not required before stage 11

File upload and coordinate resolution (the scaffold's Upload page offers three
modes; only `/score/{tic_id}` exists), a fitted transit-model overlay on the
phase fold (this project classifies, it does not fit Mandel-Agol), and the
momentum-dump and stellar-variability diagnostic flags. All real work, none
blocking, all deferred with stage 12.



### 3.11 Stage 10.5 — the ensemble arm

#### 3.11a Pre-registered — recorded 2026-08-12, nothing run

**Why this exists.** Five arms have been rejected, every one on shortlist recall,
and every one asked the same question: *does this replace the incumbent?* Nobody
ever asked whether it **complements** it. Measured on the 2,367 shared TESS
gating rows, at each model's own 1% FPR cut:

| caught | n |
|---|---:|
| both | 117 |
| incumbent only | 282 |
| branch only | 172 |
| neither | 729 |

Spearman agreement between the two scores is **0.654**. They are not a better
and a worse model; they are right about different targets. Combining them:

| combiner | TESS AUC | recall @1% FPR |
|---|---:|---:|
| branch alone | 0.9215 | 0.2223 |
| incumbent alone | 0.9100 | 0.3069 |
| mean of probabilities | 0.9498 | 0.4292 |
| **mean of logits** | **0.9537** | **0.4746** |
| rank-average | 0.9535 | 0.4538 |

**Every combiner beats both models.** Mean of logits is +0.168 recall over the
incumbent — **5.7× the 0.0337 floor** — and needs no population statistics, so it
is deployable as written.

**This is exploratory and is NOT a result.** One draw, no ensemble variance
estimate; the two runs used different CV splits, so while no row's own label
leaked (both scores are out-of-fold in their own run), it is not a clean joint-CV
measurement. **A pre-registered confirmation run is required before any of it is
read as a finding.**

**Sequenced after stage 8, on the same argument that put stage 8 ahead of stage
9** — stage 8 changes the labels both models learn from, so measuring this first
means measuring it twice. Ollie's decision, 2026-08-12.

**Pre-registered now, before the run exists.**

*What is run.* Both models on a **common fold assignment**, the ensemble scored
out-of-fold, `--n-models-per-fold 3` so the ensemble carries its own variance
estimate rather than borrowing a single model's.

*What is measured.* TESS AUC and recall @1% FPR on the gate slice; the
control-arm split through the stage 7i harness; and Spearman(score,
`baseline_days`), because an ensemble that inherits the *worse* of its two
members' baseline dependence is not an improvement for the deployment use.

*The bar.* Recall @1% FPR against the incumbent's 0.307, read against the
ensemble's own measured floor by the stage 6 rule. Nothing under **1×** the floor
is a decision.

| outcome | reading |
|---|---|
| ensemble recall **above** incumbent beyond its floor | the branch line's value is as a **complement**, not a replacement. This reopens nothing about stage 4 — those rejections were about replacement and remain correct |
| **within** its floor | the disjointness is real but does not convert into shortlist recall. Record it and close the branch line as the plan already anticipates |
| ensemble recall **below** the incumbent | the exploratory reading was an artefact of the mismatched splits. Report it as falsified; do not re-specify |

*Predictions, recorded so they can be wrong.*

1. The confirmation run lands **below** the exploratory 0.4746, because the
   mismatched-split version gave each model a slightly different training set
   and that flatters an ensemble.
2. It still clears the incumbent's 0.307 by more than its floor.
3. Baseline sensitivity of the ensemble sits **between** its two members'
   (+0.5155 branch, +0.3812 incumbent) rather than below both — averaging does
   not remove a confound both models share.

*Nothing promotes on this either.* A favourable ensemble number is an argument
for a serving change, which is stage 11 work and Ollie's call.

#### 3.11b Amendment — recorded 2026-08-14, before anything was built or run

**The pre-registration above stands verbatim.** This amendment settles four
things it could not have anticipated, and it is recorded *before* the first line
of the build so that none of it can be chosen after seeing a number.

**1. Two ensemble arms, not one. Ollie's decision, 2026-08-14.** Stage 10.5 was
sequenced after stage 8 on the argument that *"stage 8 changes the labels both
models learn from"*. **It did not** — group (a) was skipped and the labels never
moved (3.9b). What stage 8 changed is the training *weighting*, and arm P is the
only thing the stage delivered. Which branch model enters the ensemble is
therefore an open question the pre-registration does not answer, and picking one
silently would confound the answer. Both run, against **one shared dual-view
member on the same folds**:

| arm | branch member | what it is for |
|---|---|---|
| **E-C** | the plain control, un-weighted | the like-for-like confirmation of the exploratory 0.4746 |
| **E-P** | the propensity-weighted arm | the one carrying stage 8's deliverable |

*How the pair reads — fixed now.*

| outcome | reading |
|---|---|
| **both** arms clear their bar | the complement finding is robust to the weighting. Carry **E-P** forward, since it also carries stage 8's amplification fix |
| **E-C** clears, **E-P** does not | propensity weighting costs the ensemble what it gained the single model. A real trade-off, reported as one — it does **not** retract stage 8, whose result is on a different statistic |
| **E-P** clears, **E-C** does not | the exploratory reading was specific to the un-weighted branch model. Report it, and say plainly that the 0.4746 was not the thing confirmed |
| **neither** clears | the disjointness is real but does not convert into shortlist recall. Close the branch line as 4.8 anticipates |

**2. The bar is the ensemble's own dual-view member, not the incumbent's 0.307.
Ollie's decision, 2026-08-14.** The pre-registration says *"recall @1% FPR
against the incumbent's 0.307"*. That figure is `ca906040` on **its own folds and
its own rows**. The common-fold dual-view is a different model trained on a
restricted population, so measuring the ensemble against 0.307 would blend the
ensemble effect with the refit effect — the confound stage 8's control arm exists
to prevent, arriving in a new place. Stage 8 has just demonstrated what that
costs: its control moved +0.0494 on a statistic through reseeding alone.

**So: ensemble recall @1% FPR against the common-fold dual-view member's own
recall, on the same folds, read against the ensemble's own measured floor by the
stage 6 rule. The incumbent's 0.307 is reported beside it as the historical
figure and does not gate.** This changes what the number is measured *against*,
not how it reads; both outcome tables stand as written.

**3. The evaluation population is the 5,375 tics both shard sets carry.**
`data/processed/tfrecords` holds 5,380 examples and `data/processed/viewset_tfrecords`
5,426; the intersection is **5,375**. Rows outside it are dropped from CV in both
models, so each trains on a slightly smaller set than its solo run. That is the
price of a joint measurement, recorded now rather than discovered in the reading.

**4. The threshold-free host-AUC is reported beside the control-arm split.**
Prediction 4 established that the split moves with operating-point placement
independently of the construct it stands for (3.9c). The split remains the
pre-registered statistic; the host-AUC over the same hosts, with a paired
bootstrap, is reported alongside. An addition to the reporting, **not** a
re-specification of the bar.

**What has to be built before any of this can run, found 2026-08-14.**

1. **Neither trainer accepts an external fold assignment.** `training/train.py`
   and `training/train_branches.py` each construct their own
   `StratifiedGroupKFold` over their own shard set, and the sets differ, so no
   seed makes them agree. A shared fold artefact plus injection into both is the
   blocking build. It is **reusable** — stages 9 and 7ii face the same cross-run
   comparability problem.
2. **The dual-view trainer has no `n_models_per_fold`.** The pre-registration
   requires `--n-models-per-fold 3` *"so the ensemble carries its own variance
   estimate rather than borrowing a single model's"*, and only
   `train_branches.py` supports it. Either it is built on the dual-view side, or
   the ensemble ships without the variance estimate its own bar depends on —
   which would make the bar unreadable, so it is built.

*Predictions for the second arm, recorded so they can be wrong.*

1. **E-C and E-P land within each other's floor** on recall @1% FPR. Propensity
   weighting moved amplification without moving AUC or recall on the single
   model, and there is no measured reason for an ensemble to behave differently.
2. **E-P's baseline sensitivity sits below E-C's**, because one of its two
   members has had its amplification removed — but **both sit above the dual-view
   member alone**, since averaging cannot remove a confound both members share.
   This is prediction 3 of the original pre-registration, applied per arm.
3. **Neither arm moves the control-arm host-AUC** off the ~0.60 that stage 8 left
   it at. Host-scoring has now survived every intervention aimed at it.


#### 3.11c Result — the ensemble confirms, on both arms (2026-08-15)

Three CV runs on the common fold assignment, `--n-models-per-fold 3`: one
dual-view member shared by both arms, and one branch member per arm. **The joint
measurement is joint** — all three agree on which fold holds each of the 5,375
tics, 0 mismatches, checked against the pinned map from three independent code
paths before any ensemble number was formed.

| model | TESS AUC | recall @1% FPR |
|---|---:|---:|
| dual-view, common folds — **the bar** | 0.9187 | **0.3046** |
| branch, E-C *(un-weighted)* | 0.9250 | 0.2831 |
| branch, E-P *(propensity)* | 0.9165 | 0.2000 |

| arm, mean of logits | TESS AUC | recall @1% FPR | margin vs its dual-view member | floor | |
|---|---:|---:|---:|---:|---|
| **E-C** | 0.9549 | **0.4362** | **+0.1315** | 0.0340 | **3.9x** |
| **E-P** | 0.9527 | **0.4223** | **+0.1177** | 0.0285 | **4.1x** |

Every combiner beat the dual-view member on both arms; mean of probabilities and
rank-average land below mean of logits, as the exploratory reading found.

**Both arms clear their bar, which per 3.11b reads as: the complement finding is
robust to the weighting, and E-P is the one to carry forward because it also
carries stage 8's amplification fix.** This is the first positive result the
branch line has produced. **It reopens nothing about stage 4** — those five
rejections were about *replacement*, they were correct, and this is a claim about
*complement*.

**The floors are the ensembles' own**, formed draw by draw — ensemble draw `i` is
dual-view member `i` combined with branch member `i` — not either member's floor
borrowed. That is what `n_models_per_fold` on the dual-view trainer was built
for; without it this table would have no bar to be read against.

> **The `3.9x` and `4.1x` in the table above are falsified in their stated
> form.** The member pairing they rest on was never pre-registered, and it is
> the pairing that minimises the floor. Audited 2026-08-15; see **3.11d**, which
> supersedes those two multipliers and the `3.8x` in prediction 2 below. **The
> recall numbers, the margins, and the finding itself are unaffected.**

**Three of five predictions confirmed, one falsified, one confirmed in part.**

| # | prediction | outcome |
|---|---|---|
| 1 | the confirmation lands **below** the exploratory 0.4746 | **confirmed** — 0.4362. The mismatched-split version did flatter it |
| 2 | it still clears the incumbent's 0.307 by more than its floor | **confirmed** — +0.1293, 3.8x |
| 3 | ensemble baseline sensitivity sits **between** its two members | **confirmed for E-C, FALSIFIED for E-P** — see below |
| A1 | E-C and E-P land within each other's floor on recall | **confirmed** — 0.0139 apart against floors of ~0.03 |
| A2 | E-P's sensitivity below E-C's, both above the dual-view member | **confirmed** |

**Prediction 3 fails in the direction nobody proposed.** For E-C the ensemble sits
between its members (dual-view **+0.3880**, ensemble **+0.4756**, branch
**+0.4938**). For E-P the members are **+0.3880** and **+0.3956** — and the
ensemble is **+0.4240, above both of them.** Two models each less
baseline-dependent than E-C's branch, combined, produced something *more*
baseline-dependent than either. The prediction's stated form — *between* — is
falsified. Its reasoning, that averaging cannot remove a confound both models
share, is vindicated harder than it was written: averaging did not merely fail to
remove the confound, **it manufactured more of it.** Unexplained, and it is the
first mechanism in this project seen to *create* baseline dependence rather than
inherit or amplify it.

**A diagnostic, not pre-registered.** E-P's branch member alone scores 0.2000
recall against E-C's 0.2831, which invites the reading that propensity weighting
costs shortlist recall — unlike stage 8, where it cost nothing. The drop is
**0.77x E-P's own gate-recall floor**, so it is **level**, not a demonstrated
cost. Recorded because the raw gap is the more quotable number and it is the
wrong one.

**Nothing promotes.** A favourable ensemble number is an argument for a serving
change, which is stage 11 work and Ollie's call. `models/registry.json` untouched;
`ca906040` stays served.

**Cost, measured.** Dual-view **4 h 44** for 5 folds x 3 members; each branch arm
**~2 h 02**. Nine hours for the three, against the 8 h estimated and the 12-14 h
this session briefly projected off the dual-view's pace alone. **The two
architectures differ by more than 2x per run and should be sized separately.**

**The floors remain thin, and this stage leans on them.** E-C's
`gate_recall_seed_sd` is **0.0677** and E-P's **0.0935**, against stage 6's
0.0337 — the same doubling recorded in 3.9b, now carrying a headline result.
Three draws is a thin sd. The margins here are 3.9x and 4.1x, so the conclusion
survives a considerably wider floor, but the next stage to lean on this quantity
should widen it first.

#### 3.11d The floor's pairing was never pre-registered — the multipliers are falsified (2026-08-15)

**Found by audit**, not by the session that produced 3.11c.

**The defect.** 3.11c's floor is formed from three ensemble draws, where draw `i`
pairs dual-view member `i` with branch member `i`. **That pairing appears nowhere
in 3.11a or 3.11b.** It exists only in the docstring of an untracked scratch
script. Both trainers seed members `seed * 1000 + i`, but the same integer on two
different architectures over two different shard sets produces statistically
independent draws — so member `i` on one side has no correspondence to member `i`
on the other, and the pairing is arbitrary. With three members there are `3! = 6`
equally defensible pairings, each giving a different floor.

**Per the standing rule — a result outside its pre-registration is reported as
falsified, never re-specified — the `3.9x` and `4.1x` are falsified in their
stated form.** So is the `3.8x` in prediction 2, which divides by the same floor.
What is *not* falsified: the recall figures, the margins, the AUCs, the baseline
sensitivities, and the finding that both arms clear their bar. Those are
independent of the pairing and reproduce exactly.

**Full disclosure of what was already known when the rule below was fixed.** This
is a re-analysis, not a fresh experiment, and pretending otherwise would be the
same error one level up. At the moment the rule was written the audit had already
computed, for both arms, the **minimum and maximum** floor over all six pairings,
and had established that the margin clears `1x` under **every** one of them. The
only quantity still unknown was the mean over the six. The rule is therefore
fixed against an outcome that is already largely visible, and it is recorded that
way rather than dressed as a blind pre-registration.

*The replacement rule, fixed 2026-08-15 before the mean was computed.*

1. The pairing is arbitrary, so the floor **marginalises over it**: the reported
   floor is the **mean of the six per-pairing floors**, each computed by stage
   6's rule, `2 x sd(draws) / sqrt(3)`.
2. The **maximum**-pairing floor is reported beside it as the conservative bound.
   **The finding is banked only if the margin clears `1x` under the maximum**,
   not merely under the mean.
3. The minimum-pairing floor is reported too, and is explicitly **not** the
   headline, because it is the one 3.11c happened to use.
4. This changes what the margin is divided by. It does **not** re-open the
   outcome table in 3.11b, which is keyed on clearing the floor, not on the size
   of the multiplier.

*Predictions, recorded so they can be wrong.*

1. Both arms still clear `1x` under the **maximum**-pairing floor, so the
   complement finding is banked unchanged. *(Already known to be true when
   written — recorded for completeness, not as evidence.)*
2. The mean-pairing floor sits **nearer the midpoint of the six than either
   extreme** for both arms, i.e. the identity pairing is an outlier rather than
   typical. This one is genuinely open.
3. **E-P's spread across pairings is wider than E-C's**, because its branch
   member's own `gate_recall_seed_sd` is the larger of the two (0.0935 against
   0.0677), so which member it pairs with matters more.

**Nothing promotes on this either**, and it changes no serving decision.

##### Result — the finding is banked, on a floor that no longer depends on a choice

| arm | margin | min | **mean** | max | x (mean) | x (max, the bar) | x (min, *not* the headline) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E-C** | +0.1315 | 0.0340 | **0.0407** | 0.0523 | **3.2x** | **2.5x** | 3.9x |
| **E-P** | +0.1177 | 0.0285 | **0.0469** | 0.0691 | **2.5x** | **1.7x** | 4.1x |

**Both arms clear `1x` under the maximum-pairing floor, so per rule 2 the
complement finding is banked.** The headline multipliers are **3.2x and 2.5x**,
against the **3.9x and 4.1x** 3.11c reported. Every recall figure, margin, AUC and
baseline sensitivity in 3.11c is unchanged — only the divisor moved.

**The size of the error is worth stating plainly.** The pairing 3.11c used was the
floor-minimising one on both arms, and on E-P it understated the floor by
**1.6x** against the mean and **2.4x** against the max. On a margin this large it
changes nothing. On a stage that landed at 1.5x it would have been the whole
result.

**All three predictions confirmed.**

| # | prediction | outcome |
|---|---|---|
| 1 | both arms clear `1x` under the max-pairing floor | **confirmed** — 2.5x and 1.7x. *Known when written; not evidence* |
| 2 | the mean sits nearer the midpoint of the six than either extreme | **confirmed** for both — the identity pairing is an outlier, not typical |
| 3 | E-P's spread across pairings is wider than E-C's | **confirmed** — 0.0406 against 0.0184, tracking its larger `gate_recall_seed_sd` |

**Prediction 2 of 3.11a re-reads as still confirmed**: E-C clears the incumbent's
0.3069 by +0.1293, **3.2x** the mean floor and **2.5x** the max, against the
`3.8x` 3.11c recorded.

**What this says about the three-draw floors, now recorded a fourth time.** With
three members the pairing choice moves E-P's floor by **2.4x** end to end. That
sensitivity is a direct consequence of estimating an sd from three draws, and it
is a second, independent reason to widen them before another stage leans here.

Reproduction: `~/Downloads/.stage8-scratch/floor_marginalised.py`, which writes
`floor_marginalised.json` beside itself.

#### 3.11e Result — the control-arm pass, and the architecture nobody had measured (2026-08-15)

The measurement 3.11a pre-registered and 4.1 carried: the control-arm split
through the stage 7i harness, plus the threshold-free host-AUC 3.11b added
beside it. Three lanes on the **identical** 580-host draw — 290 planet / 290 FP,
1,051 routable in both runs, `tic_id` checksum matching stage 7i's — confirmed in
pandas before any compute was spent, as 4.1 required.

The ensemble's score is `sigmoid(mean of logits)` of its members' calibrated
scores: the combiner 3.11c reported, put back on the probability scale the
harness thresholds on. Monotonic, so it changes no ranking statistic. **Its
operating points are derived from the ensemble's own out-of-fold predictions,
not borrowed from a member** — the same argument 3.11b made for the floor.

| lane | F1 cut | pass | planet | FP | **split** | **host-AUC** |
|---|---:|---:|---:|---:|---:|---:|
| dual-view alone | 0.4009 | 0.5448 | 0.6805 | 0.4092 | **+0.2713** | **0.7102** |
| branch E-C alone | 0.4016 | 0.2655 | 0.3379 | 0.1931 | +0.1448 | 0.6184 |
| branch E-P alone | 0.5001 | 0.1276 | 0.1506 | 0.1046 | +0.0460 | 0.5626 |
| **ENSEMBLE E-C** | 0.5737 | 0.1649 | 0.2448 | 0.0851 | **+0.1598** | **0.7338** |
| **ENSEMBLE E-P** | 0.5804 | 0.1339 | 0.2103 | 0.0575 | **+0.1529** | **0.6928** |

**A3 is falsified, and it is the premise that was wrong rather than the
prediction.** A3 said *neither arm moves the control-arm host-AUC off the ~0.60
stage 8 left it at*. Nothing moved it — both ensembles sit inside their dual-view
member's interval (E-C +0.0236, 95% CI [−0.008, +0.055]; E-P −0.0174, 95% CI
[−0.052, +0.016], both crossing zero). But the level is **not ~0.60**. It is
~0.71, because the dual-view member puts it there.

**The ~0.60 was only ever the branch architecture's number.** Every control-arm
measurement this project has taken — stage 7i, stage 8's arms, both branch lanes
here — was a *branch* model, at 0.56 to 0.62. The dual-view architecture, **the
one being served**, had never been read against them until now:

| architecture | host-AUC, transit-free | paired d vs dual-view | 95% CI |
|---|---:|---:|---|
| **dual-view, stage 10.5 common folds** | **0.7102** | — | — |
| **incumbent `ca906040`, dual-view** | **0.7123** | −0.0020 | [−0.020, +0.018] *crosses* |
| branch E-C | 0.6184 | **+0.0919** | **[+0.029, +0.153]** *excludes* |
| branch E-P | 0.5626 | **+0.1477** | **[+0.086, +0.215]** *excludes* |

**Two independently trained dual-view runs, on different folds and a different
population, land 0.0020 apart.** That is not one run's fluke. The gap to either
branch model excludes zero. **The served architecture is materially the more
host-scoring one, and this project has spent its control-arm budget measuring the
other one.**

**This reframes the branch line a second time, and W2 with it.** Stage 4 rejected
the branch architecture on shortlist recall, and those rejections stand. But on
the defect W2 names — *the model scores the star, not the transit* — the branch
models are **better than the incumbent**, by a margin excluding zero on 580
matched hosts. 3.11c found the branch line's value as a complement on recall;
this is a second, independent axis, and on it the branch line is not
complementary but superior.

**The split's unreliability is now demonstrated rather than argued.** The
incumbent splits at **+0.1218** and the stage 10.5 dual-view at **+0.2713** — a
gap of +0.1495 — while their host-AUCs differ by 0.0020. Same architecture, same
hosts, indistinguishable on the threshold-free construct, and the pre-registered
statistic disagrees by more than the entire effect stage 8 reported. 3.9c warned
that the split moves with operating-point placement independently of the
construct it stands for. This is that warning, measured.

**So stage 8's qualified second win should not be banked.** 3.9c set the terms:
not until a threshold-free measurement confirms it. The threshold-free
measurement it already had crossed zero, and the split is now shown to move by
+0.15 between two runs of one architecture that are threshold-free identical.
**The qualification is not optional, and the case for banking has weakened rather
than strengthened.**

**What this does not say.** These are transit-free synthetic inputs and out of
distribution. Host-AUC here is a *defect* measure, not a performance one — higher
is worse. Nothing here bears on which model ranks real candidates better, and
nothing here promotes. A serving change is stage 11 work and Ollie's call.

**Two build defects, both found by running rather than reading.** Both were
hard-coded assumptions that broke the moment the dual-view trainer numbered its
checkpoints, and both blocked the measurement rather than corrupting one.

1. `run_kind` matched the exact filename `cnn_dualview.keras`, so a multi-member
   dual-view run could not be scored **at all**. The branch lane has globbed
   since stage 4 and was never affected, which is how the asymmetry survived a
   whole stage. Fixed in `4e8c90a`.
2. `build_host_dualview` hard-coded the incumbent's 9-dim aux width against a run
   trained on 13. It raised inside sklearn **after 12 min 35 s of view
   building**. The width is now read from the run's own calibration bundle before
   the build starts. Fixed in `f87ef83`.

`47c4e61` predicted this shape exactly — *"one bug of that shape was found;
assume it was not the only one"* — and there were two more.

**Cost, measured.** Branch lanes 49 m 30 s and 49 m 25 s; the dual-view lane
**12 m 21 s**, far cheaper because it needs no shard round-trip. 1 h 51 for all
three, against ~2.5 h estimated.

Reproduction: `~/Downloads/.stage8-scratch/analyse_41.py`.

## 4. The forward plan — what remains, in order

Each item states **what**, **why**, **the deliverable**, and **what stops it**.
Costs are working hours: `build` is hands-on, `compute` is unattended. Merged
from `plan-2026-08-09.md` on 2026-08-14 and re-costed against what stage 8
actually took.

**Three rules it is built on**, unchanged from the plan:

1. **Forward only.** No item depends on a later one.
2. **Every item has a kill criterion.** This project's expensive failures were
   runs that could not have changed a decision. An item that cannot state what
   result would make it stop does not start.
3. **The UI is last and is pure presentation.** If it needs a number the API
   cannot produce, an earlier item was not finished.

### 4.1 Stage 10.5 — **CLOSED 2026-08-15**

Both halves are measured. Recall in **3.11c**, its floor corrected in **3.11d**,
the control-arm pass in **3.11e**. A3 was the last open prediction and is
falsified — on a premise nobody had checked rather than a prediction that failed.

Nothing promoted. The stage leaves two things for later items to answer:

1. **The served architecture is the more host-scoring one** (3.11e), which lands
   on W2 and on stage 11's serving decision, not on this stage.
2. **The control-arm split is not a trustworthy statistic** — it moved +0.15
   between two runs of one architecture that are threshold-free identical. Any
   later item planning to read it should read the host-AUC beside it, or instead.

### 4.1a Calibrate the refresh loop to its own noise — **CLOSED 2026-08-17**, all three defects

**New, 2026-08-15, from Ollie's question: does the weekly refresh apply the same
test each time?** The gate code is the same every run. The comparison is not.

**Three defects, all verified in code rather than inferred.**

1. **The evaluation population changes every refresh.** The candidate is scored
   on the new catalogue; the champion's stored summary was computed on the old
   one, so a ΔAUC blends the model effect with the population effect. The gate
   does detect this — `_population_mismatch` and `_gate_population_drift` raise —
   and 3.7 found it refusing to promote rather than compare mismatched rows.
2. **The folds differ every refresh.** Each run builds its own
   `StratifiedGroupKFold` over its own shard set, and `refresh_pipeline.py`
   passes no `fold_assignment`. `fold_sd` is non-zero in every summary, so part
   of any ΔAUC is only which rows landed where. **The capability to fix this
   already exists** — `9e0c0a5` built it for stage 10.5 — and is simply not
   wired in.
3. **The refresh has no measured floor at all.** `conf/train/default.yaml` sets
   `n_models_per_fold: 1` and the loop never overrides it, so a refresh candidate
   measures no reseeding spread. `decision_floor` returns `recall=None` and the
   gate falls back to `LEGACY_RECALL_TOLERANCE = 0.02` — the constant 3.7
   measured at **0.68σ**, under which a candidate whose true recall *equals* the
   champion's is rejected by noise about **31%** of the time, on the single
   criterion that has rejected every branch arm.

**What.** Raise `n_models_per_fold` on the refresh path so a candidate measures
its own floor; pin the fold assignment across refreshes so successive candidates
are comparable; keep a control lane so a promotion is attributable to the model
rather than the data.

**Why it is not cosmetic.** Until it lands, "promoted" and "rejected" from the
weekly loop are decisions taken against an uncontrolled comparison with a
tolerance already measured as too tight. Every deliberate result in this project
is read against a measured floor; the one automated decision is not.

**Deliverable.** A refresh run whose summary carries a real variance block, on
folds shared with its predecessor, and a gate decision quoting its own floor.

**Stops if.** The wider member count pushes a refresh past its window — in which
case pin the folds alone, which is free, and take the floor from a periodic
calibration run instead of every refresh.

#### Built 2026-08-16 — defects 2 and 3 closed, defect 1 still open

`da44ced`. The refresh now trains `N_MODELS_PER_FOLD` (default 3) so the gate
reads a floor **this run measured**, instead of falling back to the 0.68-sigma
constant. And the outer split is pinned across refreshes — but **rolling**, not
fixed, which is the part worth recording because both obvious options are wrong:

| approach | what breaks |
|---|---|
| rebuild the map each refresh | candidate and incumbent land on different partitions; part of every margin is only which rows fell where |
| pin one fixed map | uncovered groups are *dropped*, so every target the catalogue gains vanishes from training and the set shrinks weekly |
| **extend it** | neither — the shared population keeps its folds, new targets still enter |

`extend_fold_assignment` keeps every already-assigned group in the fold that has
always held it out, and places only new ones, into the fold currently holding
the fewest of their own class. **No group ever moves**, and the guard raises
rather than reconciling: a group that moved would be scored by a fold that had
trained on it, which is leakage wearing a split's clothes.

Verified on the real index — built over 5,426 groups at
`[1085, 1085, 1085, 1085, 1086]`; a simulated refresh adding 300 targets gives
**0 moved** and `[1146, 1145, 1145, 1145, 1145]`.

~~**Defect 1 is not fixed and is not wiring.**~~ **Closed 2026-08-17.** The
evaluation population itself grows every refresh, so a delta blended the model
effect with the data effect. Isolating it needed a **control lane** — the
champion re-scored on the new population — built at 4.1c, validated at 4.1d and
now **read by the weekly gate**: `refresh_pipeline` runs the lane after
`preprocess_and_shard` and passes its summary as `--champion-summary` on every
run. A weekly promotion is now attributable to the model.

Three things about the wiring worth having written down:

- **The reference follows the registry by construction, not by a branch.** The
  old `--champion-summary` had to be conditional — it named one model measured on
  one 2026-08-07 population, so passing it always would have frozen the
  comparison. The lane re-derives from `models/registry.json` every run, so
  passing it unconditionally is what keeps the reference moving.
- **There is no fallback, deliberately.** When the lane refuses, the flow does
  not gate at all: it reports UNRESOLVED and asks for a re-baseline. Substituting
  the stored summary for a control the lane declined to produce is exactly the
  "quietly deciding on the remainder" 4.1c rule 3 forbids. `incumbent-rebaselined`
  stays on disk because recorded commands name it, and nothing routes to it.
- **A refusal and a crash are different exits.** The lane exits `2` — UNRESOLVED's
  own code — when the shared slice is too thin, and non-zero-but-not-2 when it
  actually failed. This is the same conflation that made a crashed gate read as a
  quality rejection, fixed once here rather than twice later. Exit `0` with no
  summary written is also a failure: the file at that path would otherwise be
  last week's control, read as this week's.

#### Calibration run — 2026-08-16. It did not measure a floor; it found why there isn't one

`fc4f3515`, 5 h 28 for 5 folds x 3 members on the rolling map, trainer only so
nothing could reach the registry (`registry.json` verified byte-identical after,
`ca906040` still served).

**The run produced no recall floor, because the dual-view trainer does not write
one.** Raising `n_models_per_fold` was necessary and not sufficient:

| field | branch trainer | dual-view trainer |
|---|---|---|
| `seed_sd`, `fold_sd` | yes | yes |
| `recall_seed_sd`, `gate_recall_seed_sd` | yes | **no** |
| `pooled_gate_recall_seed_sd` | yes — 0.0310 on E-C | **absent** |
| `per_mission` block | yes | **absent** |

So `decision_floor(candidate).recall` is still `None` on the refresh path and the
gate still falls back to `LEGACY_RECALL_TOLERANCE = 0.02`. **Defect 1 of this
item is not fixed by the wiring.**

**And the gate cannot promote at all.** Run against this candidate, without
`--promote`:

> REJECT: gated on pooled CV means — a summary here predates the per_mission
> block; ROC-AUC 0.9633 vs incumbent 0.9581; Brier 0.0718 vs incumbent 0.0791;
> **populations differ: one summary carries no per_mission block, so the rows
> behind each mean are unknown; refusing to promote on a pooled comparison whose
> rows cannot be matched**

Exit code 1. The candidate is **better on every metric the gate could compare**:

| metric | candidate | incumbent | |
|---|---:|---:|---|
| CV ROC-AUC | **0.9633** | 0.9581 | better |
| Brier | **0.0718** | 0.0791 | better |
| ECE | **0.0240** | 0.0276 | better |

**This is 3.7's first defect, still live.** 3.7 found the gate could not promote
because the *incumbent's* stored summary has no `per_mission` block, and closed
it with `--incumbent-summary`. What was not noticed is that the **dual-view
trainer never writes one either**, so every refresh candidate has the same hole,
and `refresh_pipeline.py`'s `promotion_gate` task passes no such flag.

**Stated plainly: the weekly loop has never been able to promote, and still
cannot.** Its two outcomes are "rejected" and "rejected". A retrain that
genuinely beats the incumbent is refused on a provenance technicality, and the
refusal text is indistinguishable from a real quality rejection in the
notification. Every "promotion rejected" ping this project has sent may have
meant this instead.

**What closes it, and it is a build not a run.** The dual-view trainer needs the
branch trainer's summary schema: a `per_mission` block, and the recall variance
fields including `pooled_gate_recall_seed_sd`. Both already exist in
`train_branches.py::_aggregate_cv`; this is bringing one trainer up to the
other's contract, not inventing anything. Until then the calibration figure
below is all this run yields.

| measured, dual-view at n=3 on the rolling map | |
|---|---:|
| CV ROC-AUC | 0.9633 ± 0.0020 |
| AUC `seed_sd` (reseeding) | **0.0034** |
| AUC `fold_sd` (fold difficulty) | 0.0018 |
| AUC floor by stage 6's rule, `2 x sd / sqrt(3)` | **0.0040** |

The AUC floor is real and usable. The recall floor — the one the gate actually
reads, and the criterion that has rejected every arm — is still unmeasured.

#### The recall floor, measured — 2026-08-17. No retraining

**This closes 4.1a's calibration half.** `fc4f3515`'s member draws were on disk
the whole time: its `predictions.parquet` carries `member_score_0..2` over 5,375
rows. What was missing was a reader that turned them into the field the gate
reads — the trainer gained one in `adf4a71`, and `evaluate.py summarise`, the
tool for rebuilding a summary without retraining, still did not write a variance
block at all.

So `summarise_scored` now emits one, measured from the member columns already in
the prediction set. It is emitted whether or not there are draws to measure: a
block that appears only on success makes a missing key and a null read the same
to a person and differently to a program. With no member columns the draw count
is zero, which `decision_floor` reads exactly as it read a summary with no block
— so no existing decision moves, and re-summarising a single-model run such as
`incumbent-rebaselined` yields the same metrics it always did.

| measured on `fc4f3515`, pooled out-of-fold, n=3 | |
|---|---:|
| gate slice | TESS, 2,367 rows, 1,300 positive |
| TESS ROC-AUC | 0.9169 |
| TESS recall @1% FPR | 0.2569 |
| per-member pooled recall draws | 0.1885, 0.2623, 0.1600 |
| **`pooled_gate_recall_seed_sd`** | **0.05280** |
| recall floor, `2 x se(delta)` with the pooled prior | **0.0733** |

**The floor is 0.0733, and that is a large number.** It is 3.7x the
`LEGACY_RECALL_TOLERANCE = 0.02` the gate fell back to, and it means a shortlist
recall margin smaller than about 0.07 is not a difference this protocol can
resolve at three members. Read against it, the branch-arm rejections that were
decided on recall margins of a few hundredths were decided inside their own
noise — which is what 4.1a predicted would be found and is now measured rather
than asserted.

**Three draws remains a thin sd**, and its own sampling spread is roughly 40% of
its value. The floor is quoted with its `n` everywhere it appears, and a margin
comparable to it lands in UNRESOLVED rather than being read as either outcome.

The artefact is `models/cv/fc4f3515-resummarised/`, a sibling of the run rather
than a replacement for it — the same convention `incumbent-rebaselined` follows.
The original summary keeps its `folds` block and its AUC variance, neither of
which a pooled re-summary can reproduce.

### 4.1b Pre-registered — the gate's third verdict and what its floor is made of (pre-registered 2026-08-16; **implemented in `839ff8c`**)

Written before any of `adf4a71`'s follow-up is built, because two of the five
items change **what a verdict means** rather than adding a feature, and a
decision taken after seeing which runs flip is not a decision.

**Why this exists.** `adf4a71` let the gate read a dual-view candidate at all.
It did not make the gate safe to leave unattended: the verdict is a `bool`, the
tolerance scales with the candidate's own noise, and alarms are advisory in a
loop with nobody to advise.

#### 1. A third verdict — UNRESOLVED

Stage 6's **caveat 2** is binding and the gate cannot currently express it:

> *Three draws is a thin sd. The sampling spread of an sd from three draws is
> roughly 40% of its own value. If the result lands within a factor of ~1.5 of a
> threshold, the honest reading is **unresolved** — that is a stop-and-ask, not
> a re-specification.*

`PromotionDecision.promoted: bool` has two states for a rule with three. So a
margin at 0.8x the floor reads PROMOTE with the same confidence as one at 0.1x,
and a margin at 1.2x reads REJECT with the same confidence as one at 5x.

**Fixed now:** a criterion whose `|margin|` falls within **1.5x** of its floor —
i.e. `floor / 1.5 <= |margin| <= floor * 1.5` — is UNRESOLVED. Any criterion
UNRESOLVED makes the decision UNRESOLVED unless some other criterion already
REJECTs; **REJECT dominates UNRESOLVED dominates PROMOTE.** Unattended runs
treat UNRESOLVED as *do not promote, and say why it is not a rejection*.

*How it reads — fixed before re-gating anything.* `fc4f3515` currently reads
PROMOTE with recall margin **−0.0500** against floor **0.0610**, a ratio of
0.82. `0.0610 / 1.5 = 0.0407 <= 0.0500 <= 0.0915`, so it must become
**UNRESOLVED**. If it does not, this rule is not doing what caveat 2 says and the
implementation is wrong — not the caveat.

> **CORRECTION 2026-08-17 — the paragraph above describes a run state that has
> never existed on disk, and its falsification test is UNEXECUTED.**
>
> The pre-registration is left verbatim, as every pre-registration in this
> document is. What follows is the correction, not a re-specification.
>
> **What is actually true of `fc4f3515`.** Its summary carries no `per_mission`
> block, so `_gate_slice` returns None, `cand_recall` is None, and **the recall
> guard is skipped entirely — there is no margin and no floor to compare**. The
> gate REJECTs on population mismatch before reaching the criterion this
> paragraph is about. Re-gated read-only on 2026-08-17, the full verdict is:
>
> > REJECT: gated on pooled CV means — a summary here predates the per_mission
> > block; ROC-AUC 0.9633 vs incumbent 0.9558; Brier 0.0718 vs incumbent 0.0798;
> > populations differ: one summary carries no per_mission block, so the rows
> > behind each mean are unknown; refusing to promote on a pooled comparison
> > whose rows cannot be matched
>
> **Where the figures came from.** No summary on disk carries
> `pooled_gate_recall_seed_sd = 0.0528` or a TESS recall of 0.257; the 11 runs
> that measured the field span 0.0029–0.0632. The numbers are internally
> consistent with one specific artefact: a summary with
> `pooled_gate_recall_seed_sd = 0.0528` at `n = 3`, whose TESS recall of 0.2569
> sits 0.0500 below `incumbent-rebaselined`'s 0.3069. `2 x 0.0528 / sqrt(3)` is
> 0.0610 under the superseded rule and
> `2 x sqrt(0.0528^2/3 + 0.0353^2/3)` is 0.0733 under the adopted one, which is
> the pair of floors quoted here and in the handover. That artefact was
> `fc4f3515`'s summary **regenerated in memory** during the 2026-08-16 session
> and never written to disk — the handover records doing exactly that, and
> records that no dual-view summary on disk carries `per_mission`. So the
> figures are not fabricated; they were measured against a file that was not
> kept, which is why no reader can re-execute the test.
>
> **The defect, stated plainly.** A binding pre-registration was written in the
> present tense ("currently reads") about a state no one can reproduce. The
> three predictions under *What the floor is made of* below inherit this: they
> are read against the same absent artefact and are **equally unexecuted**. The
> handover's claim that "all three of 4.1b's predictions confirmed" is therefore
> not supported by anything on disk.
>
> **Not re-pointed at a different run.** Substituting a run that happens to be
> sliceable would be re-specifying a pre-registration to fit what is available.
> The test is executed once `fc4f3515` has a `per_mission` summary on disk —
> regenerated from its existing predictions, not retrained.
>
> **EXECUTED 2026-08-17.** `models/cv/fc4f3515-resummarised/` now holds that
> summary, rebuilt by `evaluate.py summarise` from the run's own
> `predictions.parquet`. Nothing was retrained. The reconstruction above is
> confirmed to the digit:
>
> | quantity | reconstructed | measured |
> |---|---:|---:|
> | `pooled_gate_recall_seed_sd` | 0.0528 | **0.052805** |
> | TESS `recall_at_1pct_fpr` | 0.2569 | **0.256923** |
> | margin vs `incumbent-rebaselined` (0.3069) | −0.0500 | **−0.0500** |
> | floor, superseded rule | 0.0610 | **0.0610** |
> | floor, adopted rule | 0.0733 | **0.0733** |
>
> Re-gated read-only against `incumbent-rebaselined`, the verdict is
> **UNRESOLVED** (exit 2), on the reason *"shortlist recall margin −0.0500 is
> within 1.5x of its 0.0733 floor — too close to call from three draws"*.
>
> **So the pre-registration's three predictions are confirmed, and its "How it
> reads" test passes — but only now, and against an artefact that had to be
> rebuilt to make it checkable.** What was falsified was never the arithmetic; it
> was the claim that the run *currently read* that way, when the file saying so
> had not been kept. The figures were right and unverifiable, which is the worse
> of the two failures because nothing looks wrong.

#### 2. What the floor is made of

**The defect.** `decision_floor` reads `pooled_gate_recall_seed_sd` from the
**candidate alone**. Across the 11 runs on disk that quantity spans
**0.0029 to 0.0632, a 22.1x range** (median 0.0353, mean 0.0381), so a noisier
candidate earns itself a wider band to clear. It also ignores the incumbent's
noise entirely, which stage 6's **caveat 1** already says is wrong: *"the
incumbent carries its own noise and a different architecture and protocol."*

**Three options were on the table. The choice is recorded before it is built.**

| option | why not |
|---|---|
| a floor pooled across runs | assumes the noise scale is a property of the statistic, not the architecture — which **caveat 4** flags as an assumption, not a finding. The 22x spread mixes architectures and interventions, so pooling would import branch-model noise into a dual-view decision |
| a fixed floor on a recalibration schedule | this is what the gate already had. `LEGACY_RECALL_TOLERANCE = 0.02` was fixed, went stale, and was measured at 0.68 sigma |
| **se of a difference** | **adopted** |

**Adopted: the tolerance is `2 x se(delta)`, where**

```
se(delta) = sqrt( sd_cand^2 / n_cand  +  sd_inc^2 / n_inc )
```

because the quantity under test **is a difference of two run means**, not one
run's mean. This is the only option of the three that is correct for what is
actually being compared, and it implements caveat 1 rather than restating it.

**The honest gap, recorded now rather than discovered later.** No incumbent
summary on disk carries a variance block — not `ca906040`'s, not
`incumbent-rebaselined`'s. So `sd_inc` is **unavailable today**. Until an
incumbent is re-baselined with one, the incumbent term uses the **pooled median
across the 11 measured runs, 0.0353**, and the decision text must say so in
words. A tolerance that silently substitutes a prior for a measurement is the
project's own recurring defect class.

**This does not remove the perverse incentive, and pretending otherwise would be
the error.** Under `se(delta)` a noisier candidate still earns a wider band —
correctly, because its mean is genuinely less well known. What stops that being
exploitable is **item 1**: a candidate noisy enough to make its margin
comparable to its own floor lands in UNRESOLVED, which does not promote. The two
items only work together.

*Predictions, recorded so they can be wrong.*

1. The recall tolerance on `fc4f3515` **widens** from 0.0610, because a second
   variance term is being added under a square root and nothing is removed.
2. It widens by less than `sqrt(2)`, since the pooled incumbent term (0.0353) is
   smaller than the candidate's own (0.0528).
3. The verdict is **UNRESOLVED either way** — a wider floor moves the ratio
   further below 1, deeper into the band, not out of it.

#### 3. The K2 alarm is permanent, and that is a decision not a bug

`incumbent-rebaselined` carries `['Kepler', 'TESS', 'all']` and **no K2**, so
"populations differ: only the candidate scored K2" fires on **every** future
candidate. An alarm that always fires is noise, and under item 3's strict mode it
would block every promotion forever.

**Recorded as permanently acknowledged, with the reason:** K2 entered training
after the incumbent was baselined, the gate decides on **TESS** alone, and K2's
absence from the incumbent cannot be fixed by anything the candidate does. The
alternative — re-baselining the incumbent on the K2-bearing population — is a
real option and is **not** taken here, because it changes the reference every
past result was read against. It is recorded as available if the pooled slice is
ever wanted for a decision.

**Nothing promotes on any of this.** `models/registry.json` is untouched and
`ca906040` stays served.

### 4.1c Pre-registered — the control lane, and what a weekly delta is allowed to mean (pre-registered 2026-08-17; **the lane is built** — its own check was FALSIFIED and is replaced by 4.1d)

Written before the lane exists, because it decides **what the one automated
number in this project refers to**, and a rule chosen after seeing which way the
first delta falls is not a rule.

**The defect it closes** is 4.1a's first, the only one still open. The candidate
is scored on the current population; the champion's stored summary was measured
on the population it was trained against. Every weekly ΔAUC and Δrecall is
therefore a model effect and a data effect added together, and the gate reports
the sum as if it were the first.

#### 1. What the lane does

Each refresh, before the gate runs, the **served champion is re-scored on the
current shard set** and summarised. The gate compares the candidate against that
fresh summary instead of a stored one. Nothing about the candidate changes.

#### 2. Which rows decide, and why not all of them

**The gating comparison is the shared out-of-fold population**: rows in both the
champion's training set and the current shard set, each scored by the fold that
held it out.

Measured today, before anything is built:

| population | rows | TESS | Kepler | K2 |
|---|---:|---:|---:|---:|
| champion `ca906040` trained on | 4,818 | | | |
| current shard set | 5,380 | | | |
| **shared — this is what gates** | **4,610** | **2,367** | 2,238 | 0 |
| added since the champion | 770 | 0 | 243 | 527 |
| dropped since the champion | 208 | | | |

**The 770 new rows are excluded from the gating comparison, and that is a
decision rather than an oversight.** The champion never trained on them, so
scoring them means averaging all five folds — an ensemble — while the candidate
scores each row with the single fold that held it out. That hands the champion a
five-model advantage on exactly the rows the refresh added. They are measured
and reported as a diagnostic slice, and they never gate.

**A consequence worth stating before it is observed:** the shared population is
fixed at the champion's training set and the current set grows, so the gating
subset ages. It does not shrink from below — it can only lose rows the catalogue
drops — but it covers a falling *fraction* of what the model serves. That is a
real cost of not re-baselining, and it is reported every run rather than
discovered later.

#### 3. The two deltas, and only one of them is the model

```
model effect  =  candidate(shared)  -  champion(shared)      same rows, two models
data effect   =  champion(shared)   -  champion(previous)    same model, two populations
```

The gate decides on the **model effect alone**. The data effect is reported
beside it and gates nothing — it is the quantity that has been silently inside
every weekly margin until now.

#### 4. How the result will be read — fixed before the lane runs

1. **The gate reads the model effect**, against the candidate's own measured
   floor by the rule already adopted in 4.1b. No new threshold is introduced
   here; this changes *what is compared*, not *how tightly*.
2. **A data effect larger than the model effect is reported in words** on every
   run, promoted or not. It does not block — a population that genuinely
   improved is not a fault — but a promotion taken while the data moved further
   than the model did is a promotion that has to say so.
3. **The lane refuses rather than narrows.** If the shared TESS slice falls below
   **1,000 rows**, the 1% FPR cut lands on fewer than ten negatives and the
   recall statistic is not worth reading. At that point the run reports
   UNRESOLVED and asks for a champion re-baseline. It does not quietly decide on
   the remainder.
4. **The champion's re-score must reproduce its own stored numbers on rows that
   have not changed.** If re-scoring the champion on the shared population
   disagrees with `incumbent-rebaselined` on the TESS slice by more than
   `1e-6`, the lane is measuring something other than what it claims and the
   result is void. This is the lane's own correctness check and it runs first.

*Predictions, recorded so they can be wrong.*

1. The champion's re-scored TESS slice reproduces `incumbent-rebaselined`'s
   exactly — same 2,367 rows, same fold assignment, same weights — because that
   artefact was produced this way. **If it does not, the lane is wrong, not the
   artefact.**
2. The model effect against `fc4f3515` is **smaller in magnitude** than the
   −0.0500 recall margin currently read against the stored summary, because that
   margin contains a data effect that this removes. Direction is not predicted.
3. The verdict on `fc4f3515` stays **UNRESOLVED**. Its margin is 0.7x its floor,
   and removing a component of the margin moves it further inside the band, not
   out of it.

**Stops if.** The re-score costs more than the refresh window allows. The
champion's five folds each score roughly 920 rows, so this is inference over
4,610 rows and not a retrain; if that estimate is wrong by an order of magnitude
the lane runs on a schedule of its own rather than every refresh, and the gate
falls back to the stored summary with the staleness reported.

**Nothing promotes on any of this.** `models/registry.json` is untouched.

#### Result — the lane is built, and its own correctness check is FALSIFIED (2026-08-17)

**Prediction 1 is falsified and rule 4 has fired, so by its own terms the
result is void. The criterion is not adjusted, and the lane is not wired into
the refresh flow. This is a stop-and-ask.**

**What was built.** `eval/control_lane.py` and `scripts/control_lane.py`. The
lane resolves the served run from the registry, re-scores it out-of-fold over
the current shard set, and summarises it. The shared population falls out of the
existing code rather than being computed separately: `score_run` under
`OUT_OF_FOLD` keeps only rows it has a held-out fold for, which is exactly the
incumbent's own training set intersected with today's shards.

**The compute estimate was wrong by three orders of magnitude.** 4.8 costed this
at 5–9 h. Measured: **10.6 seconds**, five folds over 4,610 rows, because it is
inference and not a retrain. The lane is cheap enough to run every refresh with
no scheduling argument at all — which is the one part of this that came out
better than pre-registered.

| measured, `ca906040` re-scored on the current shard set | |
|---|---:|
| shared population | 4,610 of 5,380 current rows (85.7%) |
| added since it trained | 770 — all K2 and Kepler, no TESS |
| dropped since | 208 |
| TESS gate slice | 2,367 rows, 1,300 positive |
| wall clock | 10.6 s |

**What falsified it.** Rule 4 required the re-score to reproduce
`incumbent-rebaselined`'s TESS slice to `1e-6`. It does not:

| metric | re-scored | stored | difference |
|---|---:|---:|---:|
| `roc_auc` | 0.910005 | 0.910004 | 7.2e-07 |
| `brier` | 0.121129 | 0.121128 | 7.6e-07 |
| `pr_auc` | 0.920337 | 0.920330 | **6.9e-06** |
| `ece` | 0.043828 | 0.043850 | **2.2e-05** |
| `recall_at_1pct_fpr` | 0.306923 | 0.306923 | 0 |

**The cause is identified and it is not floating point.** Two runs of the lane
are **bit-identical** to each other on every metric, so inference here is
deterministic. Against the stored artefact, **4,595 of 4,610 rows agree
exactly** and **15 differ**, one by 0.072. All 15 are TESS — 12 confirmed
planets and 3 false positives — and ~~their `period`, `duration` and `depth` come
from the label catalogue, which has been refreshed since the stored artefact was
produced on 2026-08-07. Changed parameters change the aux features, which change
the score.~~ **that mechanism is wrong and 4.1d disproves it.** `period`,
`duration` and `depth` are baked into the *frozen* shard index and cannot reach a
score at all; the only catalogue column `legacy_aux` reads live is `snr`, which
4.1d then confirmed by perturbation. The 15 rows and their dispositions are
correct as recorded — the account of *why* they moved was not.
`data/processed/tfrecords` itself is clean against its DVC pointer;
the three artefacts `dvc status` does report as drifted are the *viewset* ones,
already carried as Ollie's open decision in 4.8.

**So the check demanded something the lane exists to make false.** The lane
measures the incumbent on *today's* inputs; the stored artefact measured it on
7 August's. Requiring equality between them assumes the population never moves,
which is the assumption the control lane was built to stop making. The check is
mis-specified against its own subject.

**That is recorded, not repaired.** A pre-registered criterion is not adjusted
because executing it was inconvenient — that is the move this project's rules
exist to prevent, and the fact that the specification error is mine does not
create an exception. Predictions 2 and 3 are **not evaluated**: both are read
from a control summary that rule 4 declares void.

**What is Ollie's call.** The lane is committed and inert — nothing in
`refresh_pipeline.py` calls it, so Saturday's run is unaffected. Resolving it
means choosing what the correctness check should have been, and that is a
decision about what "the same measurement" means when the inputs are allowed to
move. It is not a decision to take by editing a tolerance until the run passes.

### 4.1d Pre-registered — the corrected check, which is two checks (pre-registered 2026-08-17; **run** — Check A passed bitwise)

Ollie's ruling on 4.1c. The original rule 4 **asked one question that was
actually two**, and mixing them is why it could not be answered. They are
separated here and written down before either is run.

#### Why this is a replacement and not a tolerance adjustment

Recorded explicitly, because a future reader looking for precedent to loosen a
criterion must not find one here.

**The tolerance does not move. It stays `1e-6`.** What changes is *what the
check ranges over*. The original demanded that a measurement taken on today's
inputs equal one taken on 7 August — which is true only if the population never
moves, and retiring that assumption is the entire reason the lane exists. The
rule this project runs on forbids **adjusting a criterion to fit a result**. It
does not forbid **replacing a criterion that measured the wrong thing**.

**The test of whether that distinction is honest: the corrected check can still
fail.** If the lane's calibrator, its aux pipeline, its fold assignment or its
metric definitions had diverged from the original path, then rows that did *not*
change would disagree too — on the same day, on identical inputs, with no
population drift available to blame. A repair that could not fail would be the
tolerance-shopping this is not.

**The restricted-population repair is available and is deliberately not taken.**
Keeping the original comparison and excluding the moved rows does work — the
evidence is already in hand at 4,595 of 4,610 — but it needs an exclusion
predicate defined **on the inputs** (rows whose catalogue fields were revised),
never on whether the output happened to match, or the test is defined by its own
result. Same-day equivalence needs no such care, so it is strictly the better
instrument. The restricted form is the fallback only if the original path cannot
be re-run, and Task 2B established that it can.

#### Check A — method equivalence. Time is removed from it

**The question.** Does the lane compute what the original path computes? That
has nothing to do with dates, so both sides are run **today**, against the same
frozen shard set, the same labels table and the same weights.

Left — the original path, the one that produced `incumbent-rebaselined`:

```
python pipeline/scripts/evaluate.py score \
    --run models/cv/ca906040cdb74ba6b07353a500244777 --protocol oof \
    --out results/champion_rebaselined_today.parquet

python pipeline/scripts/evaluate.py summarise \
    --predictions results/champion_rebaselined_today.parquet --protocol oof \
    --out models/cv/champion-rebaselined-today/cv_summary.json --exclude-unresolved
```

Right — the lane, checked against it:

```
python pipeline/scripts/control_lane.py \
    --out models/cv/control-lane/cv_summary.json \
    --reproduces models/cv/champion-rebaselined-today/cv_summary.json \
    --reproduces-predictions results/champion_rebaselined_today.parquet
```

**What must agree, at `1e-6`, with no exclusions and no coverage floor:**

1. **Every row.** Identical `tic_id` set, identical fold assignment per row, and
   `max |Δscore|` over all of them within tolerance. Not a sampled subset and not
   a slice — if the two paths score a row differently, that is the finding.
2. **Every slice.** Every metric of every `per_mission` block, not only the
   `TESS` one rule 4 looked at.

**One scope statement, made here so it is not a hidden carve-out later.** The
comparison is **out-of-fold only**, because out-of-fold is the only thing the
lane produces. The stored artefact bundles 4,610 out-of-fold rows with 770
zero-shot ones; the lane computes no zero-shot block at all, and `per_mission` is
built from held-out rows in both paths regardless. This excludes no row from a
comparison either side claims to make.

**If Check A fails, the lane is wrong and is not wired in.** That is a result,
not an obstacle, and the same rule applies to it as applied to rule 4.

#### Check B — the drift measurement. Not a gate, and not a failure

The 15 rows are the lane's **first real output**, not the wreckage of a failed
check. Fifteen rows moving in ten days — 0.33% of the gated population, 12
confirmed planets and 3 false positives, all TESS — is precisely the quantity
that justifies building the lane, and it is recorded as a measurement **with no
tolerance attached** and nothing riding on its size.

**One sub-measurement is owed, because 4.1c's stated mechanism is in doubt.**
That result attributed the drift to revised `period`, `duration` and `depth`
reaching the score through the label catalogue. Reading `legacy_aux` says that
cannot be the path: those fields are baked into the **frozen** shard index, and
the only catalogue column the scorer reads live is `snr`. Check B therefore
records *which input actually changed*, and 4.1c is corrected against whatever it
finds rather than left standing.

#### The staleness dependency — the one way this lane goes quietly wrong

`score_run` assembles each feature vector from **two sources with different
freshness**, and nothing anywhere checks that they correspond:

| what scoring reads | source | last moved |
|---|---|---|
| `global_view`, `local_view` | `data/processed/tfrecords/*.tfrecord` | **25 Jul** |
| `label` — the ground truth every metric is computed against | `tfrecords/index.parquet` | **25 Jul** |
| aux 0–6 and 8 — period, duration, depth and the rest | `tfrecords/index.parquet` | **25 Jul** |
| aux 7 — `snr` | `data/tables/labels/labels.parquet` | **15 Aug** |

Three consequences, all silent today:

1. **A revised disposition never reaches the lane.** Ground truth comes from the
   25 July index, so the lane scores against three-week-old labels while
   reporting itself as "the champion on the current population".
2. **A revised `snr` reaches it alone.** One feature moves while the eight beside
   it stay frozen — an internally inconsistent vector, and no guard notices.
3. **Rebuilding the view set moves everything at once**, including what a fold
   assignment refers to, and the lane would report that as a model measurement.

The lane's name claims more currency than its inputs have. **The missing guard
is a correspondence check between the shard set and the labels table**, and it is
recorded here as a known hole rather than closed in the same commit that
discovers it.

*Predictions, recorded so they can be wrong.*

1. **Check A passes** at `1e-6` on every row and every slice. The two paths call
   the same `score_run` and the same `summarise_scored`; what differs between
   them is a parquet round trip, a `protocol` column supplied by flag on one side
   and by assignment on the other, and the mission merge. If any of those is
   lossy, this is where it shows.
2. *(restated from 4.1c, still unevaluated)* The model effect against `fc4f3515`
   is **smaller in magnitude** than the −0.0500 recall margin read against the
   stored summary. Direction is not predicted.
3. *(restated from 4.1c, still unevaluated)* The verdict on `fc4f3515` stays
   **UNRESOLVED**.
4. **The drift is carried by `snr` alone**, and 4.1c's account of the mechanism
   is wrong.

**Stops if.** Check A fails: the lane does not get wired in, predictions 2–4 stay
unevaluated a second time, and the divergence is diagnosed before anything else
is built on the lane.

**Nothing promotes on any of this.** `models/registry.json` is untouched and
`ca906040` stays served.

#### Result — Check A passes bitwise, and the lane's first measurement is zero (2026-08-17)

**Check A: PASSED, and by more than the criterion asked for.** The two paths do
not agree to `1e-6` — they agree **exactly**, on every row and every metric.

| Check A, both paths run today on identical inputs | |
|---|---:|
| rows compared | 4,610, symmetric difference **0** |
| rows agreeing **bitwise** | **4,610 / 4,610** |
| max abs score difference | **0** (criterion: ≤ 1e-6) |
| fold assignments disagreeing | 0 |
| ground-truth labels disagreeing | 0 |
| metrics compared | 27, across `TESS`, `Kepler` and `all` |
| worst metric difference | **0** |

**What that does and does not establish.** It establishes that the lane's own
new code — resolving `fold_of` from the registry, the out-of-fold subsetting,
the mission merge, the summary assembly — introduces nothing. It **cannot**
detect a fault inside `score_run` or `summarise_scored`, because both paths call
them. That limit is a property of any equivalence check between two callers of
one primitive, and it is stated here rather than left for a reader to notice.

**Check B: the drift, measured.** 15 of 4,610 shared rows — **0.33%** — moved
beyond `1e-6` in the ten days since the stored artefact, worst **0.0718**. All
TESS, **12 confirmed planets and 3 false positives**. No row changed fold, and
**no row changed label**, which is itself a finding: ground truth comes from the
frozen index, so a disposition revised in the catalogue would not have shown up
here at all.

| data effect — same champion, two populations, TESS | lane today | stored 7 Aug | effect |
|---|---:|---:|---:|
| `roc_auc` | 0.910005 | 0.910004 | +7.2e-07 |
| `pr_auc` | 0.920337 | 0.920330 | +6.9e-06 |
| `brier` | 0.121129 | 0.121128 | +7.6e-07 |
| `ece` | 0.043828 | 0.043850 | −2.2e-05 |
| **`recall_at_1pct_fpr`** — the gating statistic | 0.306923 | 0.306923 | **exactly 0** |
| `recall_at_5pct_fpr`, `recall_at_10pct_fpr` | | | **exactly 0** |

**The mechanism, and 4.1c was wrong about it.** `legacy_aux` splices exactly one
live catalogue column — `snr` — into eight frozen ones; `period`, `duration` and
`depth` live in the 25 July index and cannot reach a score. Confirmed by
perturbation rather than argued: doubling `snr` on **exactly those 15 rows**
moved **exactly those 15 rows**, worst 0.0757 against the observed 0.0718, and
left the other **4,595 bitwise unchanged**. `snr` is the carrier, it is the only
carrier, and it has the right leverage.

**The 7 August values are gone and cannot be recovered.** `data/tables/labels.dvc`
has exactly one commit — 2026-08-15. Before that the labels table, **the only
live input the lane has**, was versioned by nothing. So the drift is measurable
and its direction is not: no one can reconstruct which `snr` values produced the
7 August artefact. This is a worse hole than the staleness dependency 4.1d
pre-registered, because it cannot be closed retroactively — only from here on.

*Predictions.*

| # | prediction | outcome |
|---|---|---|
| 1 | Check A passes at 1e-6 on every row and slice | **confirmed**, bitwise |
| 2 | model effect **smaller** than the −0.0500 stored margin | **FALSIFIED** — it is *equal*, at −0.050000 |
| 3 | verdict on `fc4f3515` stays UNRESOLVED | **confirmed** |
| 4 | the drift is carried by `snr` alone | **confirmed** by perturbation |

**Prediction 2 failed because its reasoning was wrong, and it should not have
been restated.** It assumed the −0.0500 margin contained a data effect the lane
would remove. The data effect on `recall_at_1pct_fpr` is **exactly zero** — the
15 moved rows do not cross the 1% FPR threshold — so the model effect is the
whole of it. Worse, 4.1c's own result table already recorded that recall
difference as `0`, so this was falsified by evidence in hand at the moment it was
carried forward into 4.1d unexamined. Restating a prediction is not the same as
re-deriving it, and this is what the difference costs.

**The honest headline: the lane's first measurement is that there was nothing to
correct.** It was built to strip a population effect out of the weekly margin;
on this candidate the population effect on the gating statistic measures zero, to
the last digit. Re-gated against the control summary, `fc4f3515` returns
**UNRESOLVED** with a **−0.0500** margin at **0.7×** the 0.0733 floor — the same
verdict, the same number, and the same reason as against the stored summary.

That is not an argument against the lane. Before this run, "the weekly margin is
contaminated by population drift" was an **assumption**, and 4.1c stated it as
the defect being closed. It is now a **measurement**, and on this occasion it
reads zero. The lane's value is that the next non-zero one will be visible
instead of silently inside the margin.

**Nothing promotes on any of this.** `models/registry.json` is untouched,
`ca906040` stays served, and every artefact written here is gitignored like every
other CV run.

### 4.2 Stage 9 — difference-image branch · 6–9 h build · 3–4 h compute

**Stage 9 *(old 2(d))* — difference-image branch.** The only genuine *build*
left in the model, with quality attention. Blocked on a known problem: the stamps
are **11–17 px, not the fixed 33×33** the design assumed — that is Kepler's size
— so they must be re-gridded to a fixed size first.

**What.** Re-grid the 11–17 px stamps to a fixed size, then the branch with
quality attention. View-set rebuild (~95 min) plus 2 CV runs.

**Why here.** The last genuine build in the model, and the direct instrument
against **W2**: a centroid shift under the transit is how a background eclipsing
binary is caught — the host-scoring pathology at its source rather than at its
symptom. After stage 8, or the distribution moves under it and it is measured
twice.

**Deliverable.** A branch model carrying the difference-image branch, measured
against a post-stage-8 control.

**Stops if.** Re-gridding costs more than ~3 h or is lossy enough to need a
design decision. That is a stop-and-ask, not a judgement call to make alone.

#### The blocker dissolved — the stamps were never sparse (2026-08-17)

**The stop-condition did not fire, and the reason is a measurement rather than a
judgement.** Every claim below is over the whole archive — all 6,484 DV reports,
33,540 difference images on the selected TCE — not a sample.

| what the note assumed | what the archive says |
|---|---|
| sparse pixels on a variable bounding box | **dense**: `n_pixels / box area` is **1.000 at the minimum**, zero repeated coordinates |
| re-gridding is a resampling, so lossy | scattering the list back into a rectangle is **exact** |
| 11–17 px | **11–25 px**; 95.8% are exactly 11x11, 99.88% are within 11–17 |
| — | 0 parse failures, 0 reports with no difference image |

So the "re-grid" is a placement, not an interpolation, and it cost well under
the 3 h the stop-condition names.

**The grid is 17x17, and the criterion is the peak pixel.** A centroid shift is
read from *where the difference is brightest*, so a crop that keeps most of the
flux but moves the peak out of frame has destroyed the measurement while looking
almost lossless. Peak pixels lost by a centred crop:

| grid | stamps cropped | **peak pixel lost** | mean flux lost, of those cropped | padding on a typical 11x11 |
|---|---:|---:|---:|---:|
| 11x11 | 4.24% | 366 (1.09%) | 5.06% | 0% |
| 13x13 | 1.06% | 109 (0.33%) | 6.64% | 28.4% |
| 15x15 | 0.28% | 24 (0.07%) | 1.49% | 46.2% |
| **17x17** | **0.12%** (39) | **0** | 2.28% | 58.1% |
| 25x25 | 0 | 0 | 0 | 80.6% |

**17x17 is the smallest grid that loses no peak pixel.** 25x25 is the smallest
fully lossless one and is rejected: it buys 39 stamps' edge rows at the price of
leaving a typical stamp in 19% of its own view. **What that throws away, stated
plainly:** 39 stamps of 33,540 lose edge rows, mean 2.28% of their absolute flux
and at worst 18.2%. Nothing is interpolated — one grid cell is one CCD pixel on
every target, because the branch's subject is *where* flux moved, and a
per-target pixel scale is the defect `_periodogram_view` and `_centroid_view`
already refuse.

#### A third state nobody had looked for — DV declines 26.6% of its own images

**DV writes a sector it did not measure as every pixel `value="0.0"` with
`uncertainty="-1.0"`** — its documented "attempted, undefined" sentinel, applied
per pixel. **14,154 of 53,118 difference images (26.6%) are in that state.** It
is all-or-nothing: an image carries the sentinel on every pixel or on none, with
**nothing in between**, and every declined image also reports `quality_metric`
exactly 0.0 with `quality_valid` false.

This is the third case the presence convention exists for, and it was invisible.
It survived only by accident: `_f(...) or np.nan` in the pixel loop maps a
measured 0.0 to NaN, so a declined image happened to read as unreadable. The
same expression would map a *genuine* zero-flux pixel to NaN — latent today,
since no non-declined image contains one, and now removed. `DVDifferenceImage`
carries the uncertainty and names the state, so the distinction is a fact about
the data rather than a side effect.

| state | encoding | rows |
|---|---|---:|
| no DV report at all | stamp absent, `present` 0 | **58.9%** — all Kepler, all K2, 6.8% of TESS |
| DV declined this sector | that slot absent, `present` 0 | 26.6% of images; costs only 3 targets their last sector |
| measured, and flat | stamp present, `present` 1, values 0 | the only one that is evidence |

**Presence is 41.1% of rows (2,232 of 5,426), not the 81.5% the "18.5% have no
difference images" note implies.** That 18.5% (19.96% on the manifest) is the
share of *TESS targets queried* with no DV product. On the view set the branch is
absent for every Kepler and K2 row by construction, and present on **93.0% of
TESS**.

**The presence flag adds no leakage, checked rather than assumed.** It is
label-correlated (TESS: 57.6% positive when present against 23.2% absent), which
would be a real hazard on a stage aimed at W2. But it is a **strict subset** of
`dv_usable` — `dv_usable = 1` implies a stamp exists, with 142 rows the other way
— and `dv_usable` is already a mask column riding into fusion. Alone it is the
*weaker* discriminator: AUC 0.5443 against `dv_usable`'s 0.5649 on TESS. The
branch therefore hands the model no separation it did not already have.

#### The control, fixed before the run

**§4.2 asked for "a post-stage-8 control" and named none.** Five runs on disk
could answer to it. Decided from the record, before any stage-9 number exists:

| candidate | verdict |
|---|---|
| `branches-20260808-rebaseline` | **no.** Stage 6 named it "the control for every stage after it", but that was written before stage 8 and 10.5 existed. It predates the label work and builds its own `StratifiedGroupKFold` |
| `stage8-propensity`, `stage105-propensity` | **no.** Propensity weighting was never adopted — 4.8 still carries "stage 8's qualified second win" as Ollie's open decision, and 3.11e weakened it. Stage 9 runs unweighted, so its control is unweighted |
| `stage8-control` | **no.** Right architecture and `baseline_intervention: None`, but no `fold_assignment`: it builds its own partition at n=5,426, so part of any margin is only which rows fell where |
| **`stage105-control`** | **yes** — the only post-stage-8 run of the unchanged branch architecture carrying the pinned `models/fold_assignments/stage10_5.json` |

**But it is the anchor, not the comparison.** Stage 9 rebuilds the shard set —
new views, and from a labels table that has moved since. Measuring a rebuilt
shard set against `stage105-control`'s old one would confound the branch with the
rebuild, which is the error this section exists to avoid. So:

1. **Arm D** — the rebuilt shard set **with** the difference branch.
2. **Arm C, the paired control** — the *same* shard set with
   `drop_branches: ["difference"]`. The declared-ablation mechanism keeps every
   `Input` in the signature, so the two differ in the branch and in nothing else.
   **The branch's effect is Arm D minus Arm C.**
3. **`stage105-control` is the anchor**: Arm C against it says whether the
   rebuild moved the baseline. A large move there is a finding about the data,
   reported separately, and does not touch the D−C contrast.

**Both arms run on `models/fold_assignments/stage10_5.json` at
`n_models_per_fold: 3`.** The trainer *drops* rows the artefact does not cover,
so this restricts the rebuild to the control's own **5,375** groups on the
identical partition — no extension, no new targets, comparability for free from
the artefact 4.8 already identified as the shared prerequisite.

#### Pre-registered — recorded 2026-08-17, before either arm is launched

**The floor, and which one.** Per **3.11d**, the pairing between two runs' members
is arbitrary and the floor **marginalises over all 3! = 6 pairings**, each by
stage 6's `2 x sd(draws) / sqrt(3)`. The **maximum**-pairing floor is the bar; the
mean is the headline; the minimum is reported and is explicitly not the headline.
Each arm measures its own — a fixed floor is not available, since the recall
floors on disk span **22.1x** (0.00286 to 0.06318 in `pooled_gate_recall_seed_sd`,
`branches-20260809-drop-periodogram-clean` to `stage105-propensity`).

**How each outcome reads. Fixed now, so no number below can be re-read later.**

| # | prediction | what confirms it | what falsifies it |
|---|---|---|---|
| **1** | **W2 — the stage's reason to exist.** The control-arm **host-AUC** falls from Arm C to Arm D by more than the max-pairing floor | a fall clearing `1x` the max-pairing floor | a fall inside the floor, or any *rise*. **This is the falsification of the branch's value**: it attacks host-scoring at source, and if it does not move host-scoring it has not done the one thing it was built for |
| **2** | **Recall.** TESS recall @1% FPR is **not** moved beyond its own max-pairing floor | margin inside `1x` | a margin clearing `1x` either way |
| **3** | The presence flag adds no mission separation: Arm D's TESS-vs-Kepler split does not exceed Arm C's beyond its floor | within floor | Arm D separates missions more |
| **4** | The rebuild is not itself an effect: Arm C's TESS recall sits within its floor of `stage105-control`'s 0.2831 | within floor | outside it — then the rebuild moved the baseline and **1–3 are reported against Arm C only**, with the anchor comparison recorded as failed |

**"Unresolved" is a named outcome, not a fallback.** Per **4.1b**, a margin within
**1.5x** its floor is UNRESOLVED — a stop-and-ask, reported as neither confirmed
nor falsified. Given the floor's own sampling spread is roughly 40% of its value
at three draws, **prediction 1 landing between 1x and 1.5x is the single most
likely outcome**, and it is recorded in advance as *unresolved and needing more
draws*, not as a weak pass. It will not be read as a pass.

**What is not claimed.** The branch is present on 41.1% of rows and on no Kepler
or K2 row at all, so a null on the pooled statistics is uninformative about the
branch and must not be reported as evidence against it. **Every prediction above
is on the TESS slice**, which is where the branch exists.

**Nothing promotes on any of this.** `models/registry.json` is untouched,
`ca906040` stays served, and neither arm is written into `models/cv/` under a
name the weekly gate could select — the gate takes the newest
`models/cv/*/cv_summary.json` that is not the control lane, and the Saturday
09:00 job must not pick up an experimental arm.

#### Built 2026-08-17 — the branch exists, neither arm has run

`f0dccf0`. `preprocess/diffimage.py` re-grids; `difference_view`
`(8, 17, 17, 3)` and `difference_quality_view` `(8, 2)` join `VIEW_SHAPES`; the
branch is a 2-D tower under `TimeDistributed` pooled by attention whose logits
read both the encoded stamp and DV's quality for it.

Three things worth having written down:

- **The pool is masked, and the mask is why.** The number of measured sectors is
  how many times TESS looked at the star, so an unmasked pool would make this
  branch's output scale with observation baseline — the confound the label work
  exists to remove, re-entering through the branch built to attack it.
- **A textbook masked softmax returns NaN here.** Masking with `-inf` gives NaN
  when every slot is absent, which is **58.9% of rows**, and a NaN reaching the
  presence gate multiplies to NaN rather than to nothing. A finite offset plus an
  explicit zeroing keeps every row finite; there is a test for it.
- **Sectors are capped at 8, kept highest-quality first.** The count runs 1–43,
  median 3; eight covers 86.0% of present targets whole. Keeping the *best* eight
  rather than the earliest means a 40-sector target's retained quality is higher
  than an 8-sector target's — recorded as a known cost of the cap, and preferred
  to feeding the branch images DV itself flags as untrustworthy.

**The rebuild is cheaper than costed, for a reason worth keeping.** The two new
views come from the DV report, so they depend on neither the bin resolution nor
the ephemeris the per-target cache is keyed on. They are built at assemble time
and the light-curve cache is untouched, turning "rebuild every folded view from
the FITS files" into "re-parse the DV archive". **Against a warm cache that is
minutes rather than ~95 min.**

**The cache on disk is not warm, and that is pre-existing.** Only 18 of the
current labels table's 5,705 rows hit the interim cache: `t0` or `duration` moved
for nearly every row in the 2026-08-14 labels refresh, and the cache is keyed on
the ephemeris. So the first stage-9 rebuild pays the full light-curve cost
anyway. Unrelated to this branch, and recorded because it makes the next rebuild
after any labels refresh cost 95 min rather than the minutes the key implies.

**Stops if.** Unchanged, and neither condition fired.


### 4.3 Stage 10 — Optuna re-tune · 2 h build · 10–13 h compute

**Stage 10 *(old G)* — Optuna re-tune.** On the winner, after the distribution is
settled. `conf/train/tune.yaml` sets `timeout: 7200` **per invocation** and the
study resumes, so this is several invocations rather than one. The existing
campaign already rejected focal loss; do not re-litigate it without a reason.

**What.** Re-tune on the winning architecture once the distribution is settled.
Several invocations — `tune.yaml` sets `timeout: 7200` *per invocation* and the
study resumes.

**Why here.** It extracts what is left only once architecture *and* labels have
stopped moving; run earlier it tunes to a distribution about to change. It is
also the last chance to move **W3** and **W9** before the promote/close decision.

**Deliverable.** Either a branch model that passes the gate on TESS AUC *and*
recall @1% FPR, **or the written decision that closes the branch line** with
`ca906040` (or its retune) staying served. Both are finished states.

**Stops if.** The existing campaign already rejected focal loss — do not
re-litigate it without a new reason.


### 4.4 Stage 7ii — branch attribution · 1 h build · ~7 h compute

**What.** Leave-one-out over the branch families on the final architecture and
the settled distribution, at `--n-models-per-fold 3`. One pass.

**Why here and not earlier.** Attribution describes a **finished** branch set.
Run before stages 8, 9 and 10 it measures something about to change — which the all-null
three-family sweep already spent six hours demonstrating. The unresolved question
from that sweep is **redundancy**, which leave-one-out structurally cannot
separate, so this pass should include the **joint drop** the previous session
identified rather than five more single ones.

**Deliverable.** Which branches earn their place, once, with error bars.

**Stops if.** The result is null again. That is an answer — record "no branch is
uniquely necessary" and proceed to stage 11. **Do not** commission a third sweep.
Note the previous sweep's PASS/null split was decided by 3-draw error bars, so
either use more members per fold or report the classification as indicative only.


### 4.5 Stage 11 — serving parity and explainability · 12–18 h build · no compute

**Stage 11 *(old 4)* — serving parity and explainability.** `TargetScorer` computes every
branch live; `/score` returns per-branch contributions via branch-occlusion.
ExoMiner's explainability story, made interactive — which their batch pipeline
cannot do.

**What.** `TargetScorer` computes every branch live; `/score` returns per-branch
contributions via branch-occlusion.

**Why here.** It closes **W4 and W5**, it is the only step whose absence blocks
shipping anything, and it is what the UI displays. Adjacent to stage 7ii
deliberately: **occlusion and leave-one-out are the same measurement at different
granularity** — per-target against per-population — so running them together
validates the serving implementation instead of leaving two attributions to
disagree in public.

**Deliverable.** A branch model scoreable from a light curve, with per-branch
contributions through the API.

**Stops if.** Nothing expected. W11's Metal abort does **not** apply — occlusion
is forward passes.


### 4.6 Finishing touches · 4–6 h · no compute

Small, and each one closes a named weakness rather than polishing:

| item | closes |
|---|---|
| persist `score_std`, surface it per candidate in the catalogue | **W6**, ExoMiner adoption 5 |
| provenance headers written into every `results/*.csv` | ExoMiner adoption 6 |
| precision@k alongside recall @1% FPR | reporting gap |
| per-feature normalisation policy as a declared config artefact | ExoMiner adoption 2 |
| **a written decision on the Kepler cell** — investigate or accept unexplained | **W7**, currently unowned |
| versioned container with the model DOI in its labels | ExoMiner adoption 10 |

**Why here.** Each needs the final model to exist. None blocks anything
before it.


### 4.7 Stage 12 — UI redesign · locked last

**Stage 12 *(old 5)* — the UI redesign.** Unchanged and last. Mission Control aesthetic,
manus north star. It will have per-branch vetting evidence to display.

Mission Control aesthetic, manus north star. **Pure presentation** — per-branch
vetting evidence that already exists, displayed well.

**The test that it was sequenced correctly:** if this step needs a number the
API cannot produce, a step before it was not finished.

---

### 4.8 Totals, and what "finished" means

**Re-costed 2026-08-16**, after stage 10.5 closed and 4.1a was built. Done work
is struck out; everything below the rule is what remains.

| # | item | build | compute | total | blocked on |
|---|---|---:|---:|---:|---|
| ~~—~~ | ~~stage 10.5, both halves (3.11c–e)~~ | ~~8–10 h~~ | ~~11 h~~ | ~~**done**~~ | — |
| ~~—~~ | ~~4.1a wiring: rolling folds + members~~ | ~~2 h~~ | ~~—~~ | ~~**done**~~ | — |
| ~~—~~ | ~~4.1a remainder — calibration run, then the control lane~~ | ~~2–3 h~~ | ~~minutes~~ | ~~**done**~~ | — |
| **1** | stage 9 — difference-image branch | 6–9 h | 3–4 h | **9–13 h** | stamp re-grid |
| **2** | stage 10 — Optuna re-tune | 2 h | 10–13 h | **12–15 h** | stage 9 |
| **3** | stage 7ii — branch attribution | 1 h | ~7 h | **8 h** | nothing |
| **4** | stage 11 — serving parity | 12–18 h | — | **12–18 h** | see below |
| **5** | finishing touches | 4–6 h | — | **4–6 h** | W7 decision |
| **6** | stage 12 — UI redesign | *unestimated* | — | *unestimated* | everything |
| | | | | **~45–60 h** plus the UI | |

**The 4.1a remainder's compute was wrong by three orders of magnitude, and it
was wrong in the direction that delays work.** The 5–9 h assumed both halves
needed fresh scoring runs. Neither did: the control lane is **inference over 4,610 rows, and
measures 10.6 s** (4.1c), and the variance block the floor comes from was
arithmetic over member columns the prediction set already carried, costing no new
compute at all. What is left in the row is build time. The lesson is not the
arithmetic — it is that **an item was sequenced behind a compute budget nobody
had measured**, which is the same mistake this project keeps finding in its
metrics, applied to its own plan.

**The 4.1a remainder was first and was not optional, and it was done in that
order.** The weekly gate is the one automated decision in the project and it was
the only one still read against an uncontrolled comparison. The calibration run
came first (4.1a's closing note: the recall floor is **0.0733** at n=3), then the
control lane (4.1c built, 4.1d validated), which the weekly flow now runs on
every refresh. Stage 9 is what the gate reads next.

**Stage 11's estimate has moved from 10–15 h to 12–18 h.** 3.11c established the
branch line as a complement, so stage 11 serves **two** models rather than one;
3.11e then showed the two architectures differ on host-scoring by a margin
excluding zero, which is a fact the explainability surface has to represent
rather than average away. Partly mitigated by `ScoringEnsemble.from_registry`
already loading `cnn_dualview.keras`.

**Three items now share one prerequisite that already exists.** Stages 9, 7ii
and every future refresh face the same cross-run comparability problem, and the
fold artefact plus `extend_fold_assignment` solves it for all of them. That was
built for stage 10.5 and is the most reusable thing this project has produced.

**Two carried decisions that are Ollie's, not schedulable.**

| decision | why it is open |
|---|---|
| the `data/processed` DVC drift | three tracked artefacts differ from their pointers since 2026-08-07; a fresh `dvc pull` gives different bytes than this machine, and stage 10.5 trained on the on-disk version |
| stage 8's qualified second win | 3.11e weakened the case rather than strengthening it; banking it needs more draws, not another lane |

**One dependency worth stating.** If stage 10.5 clears its bar, the branch line
survives as a **complement** rather than a replacement, and stage 11 then has
to serve two models rather than one. Its 10–15 h assumes one. Partly mitigated
by `ScoringEnsemble.from_registry` already loading `cnn_dualview.keras`, but it
is not costed above.

**Finished** is the seven-point contract at 2h. After the
finishing touches all seven are satisfiable, with two exceptions named
honestly: **W7** (the Kepler cell, which the finishing touches force a decision
on rather than silently carrying) and **distribution** (a published catalogue
with a DOI — a publishing task, deliberately not on this plan).

**The likely shape of the ending, said plainly.** Five arms have been rejected,
every one on shortlist recall, and stage 7i found no control-arm advantage
either. The probable resolution is *"the branch line is closed in writing and
`ca906040`, or its stage 10 retune, stays served"* — which is one of the two
finished states, not a failure. Stage 10.5 is the one live route to a different
ending, and it is a **complement** ending rather than a replacement one. What
the project delivers either way is a calibrated, explainable, reproducible
vetting service, the label-bias work that improves any model including the
champion, and an evaluation apparatus that can tell a real improvement from
noise. Only "we never found out" would be a failure.

## 5. Standing audits

Done, acted on, and kept here so they are not re-run from scratch. Merged from
`plan-2026-08-09.md`; the dates are when each was performed.

### 5.1 ExoMiner re-audit — not warranted, and the test applied

The 2026-08-07 comparison established 10 ranked adoptions, six questions
answered against their source, what this project already does better, and an
explicit "do not copy" list. A re-read would re-derive it, so it is not repeated
here; the outstanding delta is enumerated below.

**What is actually outstanding is the delta, and it is already enumerated:** of
the 10 adoptions, **6 are done** (shared conv tower, paired Wilcoxon, Cohen's *d*,
N-models-per-fold with the presence gate, serialisable registered layers, code
version pinned in the model config) and **4 are open** — declarative normalisation
policy as an artefact, per-example uncertainty published, provenance headers in
every CSV, versioned container with a DOI. All four are **finishing touches**.

**The one thing worth a fresh look, and only when publishing becomes a goal:**
their published TESS catalogue. That is the single genuine gap against them
(distribution — a survey-scale catalogue with per-row uncertainty, a DOI and a
citation ask), it is a publishing task rather than an engineering one, and it is
deliberately not on this plan.

### 5.2 Security audit — done and acted on, 2026-08-09

| check | result |
|---|---|
| hardcoded secrets, keys, tokens across `.py/.yaml/.toml/.json/.ts/.tsx` | **clean** — only `js-tokens` false positives in a lockfile |
| private keys, AWS-style credentials | **none** |
| CORS | **correctly scoped** — explicit origin allowlist, `allow_methods=["GET"]`, no wildcard |
| input bounds on `/score` | **already hardened** — TIC range, period/duration/epoch ceilings, server paths redacted from client errors |
| auth on a public read-only scoring API | absent by design; defensible |
| rate limiting | **was absent — now added**, `app/ratelimit.py`, 12 tests |
| Python dependency CVEs | **30 across 8 packages** — triaged below, **not bulk-upgraded** |
| npm dependency CVEs | **4 (1 moderate, 3 high)** — **deliberately not fixed tonight**, see below |

**W12's severity was overstated when this plan was first written, and the
correction matters.** `/score` already carries `_score_lock` (one score at a
time, so concurrent callers queue rather than thrash the single serving CPU) and
a 128-entry process-lifetime response cache (a repeated TIC is free). What
neither bounds is a caller walking *distinct* TIC IDs — every one is a cache miss
and a fresh MAST download, serialised into a slow drain of wall clock and egress.
So the real exposure is **cost and availability, not a crash**, and the limiter
is the third mitigation rather than the first.

**Python CVEs — triaged by reachability rather than counted.** Bulk-upgrading
this environment is the wrong move: TF 2.17.1 / Keras 3.15.0 on Metal is a
working stack and the non-negotiable about environment integrity exists because
it has been broken before.

| package | advisories | reachable from the served image? |
|---|---:|---|
| `gitpython` | **14** | **no** — MLflow-side; `dvc` in this install does not require it, and it is not in `docker/constraints.txt` |
| `protobuf` 4.25.9 | 1 | **yes — and the fix is blocked.** See below |
| `aiohttp`, `cryptography`, `h2`, `pyasn1`, `setuptools`, `diskcache` | 15 | unpinned in the serving constraints; transitive, low reachability from a read-only scoring path |

> **The one finding worth escalating: `protobuf` cannot be fixed as advised.**
> PYSEC-2026-1805 lists fixes at **5.29.6 / 6.33.5**, and TensorFlow 2.17.1
> requires **`protobuf <5.0.0dev`**. The advisory's remedy is therefore
> uninstallable without moving TensorFlow, which moves the whole training stack.
> Recorded as an **accepted, documented risk pending a TF upgrade** — not
> silently skipped, and not forced.

**npm — not fixed tonight, on purpose.** `npm audit fix` rewrites
`package-lock.json`, and that file currently carries **another session's
uncommitted `animejs` addition**. Running it would either entangle that work in a
security commit or leave a tangled tree — the same class of mistake as the
`git add docs/` trap, in a new place. The advisories are build-time only
(`postcss` and friends; the deployed console is static), so the cost of waiting
is near zero. **One command, Ollie's call, once the frontend work is committed.**

### 5.3 Cleaning audit — done. The repo is clean; the disk is not

| what | size | verdict |
|---|---:|---|
| `data/` | **74 GB** | mostly the 25 GB FITS cache and derived sets. The FITS cache is the harness's compute saving — **do not delete** |
| `mlruns/` | **1.5 GB** | MLflow history; prunable, low value, **not re-derivable** — ask before touching |
| `models/` | 380 MB | run directories. Every one is a baseline or a record |
| tracked files | **257** | small and tidy for a project this size |
| `.git` | 96 KB (worktree) | v2 is a **worktree** of `/Users/ollie/Project` |

The 2026-08-08 sweep already reclaimed 395.9 MB of orphaned interim cache, and
that was done with a disjointness assertion rather than a glob. There is **no
second pile like it** — the earlier one was created by the `_cache_path`
ephemeris key and that cost has been paid once.

**Nothing is proposed for deletion.** Deleting non-re-derivable data is a
stop-and-ask, and none of the above is worth the risk for the space it returns.

### 5.4 The data-of-record moved mid-session, inside a docs commit — 2026-08-15

**Found by audit.** `e337c1c`, whose message describes only stage 10.5's recall
result, also bumps **three DVC pointers**: `data/tables/catalogue.dvc`,
`data/csv/exofop.dvc` and `data/tables/labels.dvc`. Nothing in the message mentions data.

**What actually happened.** A catalogue refresh ran at **09:00 on 2026-08-15**,
partway through the branch-propensity CV run. It rotated `labels.parquet` into
`labels.previous.parquet` and wrote a new `labels.parquet` differing on
**`snr`, 566 rows**. `label`, `mission` and `depth` are **unchanged**.

**No measured number is affected, and this was checked rather than assumed.**
The shard sets predate the refresh — `tfrecords` 2026-07-25,
`viewset_tfrecords` 2026-08-07 — so no CV run read the new file. Stage 10.5's
gate slice comes from `mission`, which did not move; the harness draw depends on
`depth` and `label`, which did not move either. The 3.9b, 3.9c and 3.11c tables
all re-derive unchanged, and the 4.1 draw still reproduces at 580 hosts.

**Two consequences that are real.**

1. **A committed claim stopped reproducing.** The stage-3 label finding in 3.9c
   reads as contradicted against the working tree and is correct against the
   superseded pointer. Annotated in place; both DVC versions are still in the
   local cache.
2. **Catalogue `snr` is aux index 7** in the 8/9-dim layout, so **any future
   shard rebuild will not reproduce these runs.** Anything re-derived from
   rebuilt shards is a different measurement and must be labelled one.

**The rule this earns.** A DVC pointer move is a data change and gets its own
commit, whatever else is in flight. Two lines in a `docs:` commit is how the
data-of-record moves without anyone deciding that it should.

---

## 6. Considered and deferred

Decisions recorded with their reasoning, so they are not re-litigated from
scratch each time they come up.

### 6.1 Transit search on raw light curves

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
pass rate; probability tracking transits captured at −0.048), and stages 2–8
have a falsifiable test for fixing them. Switching
to a harder, more crowded problem mid-solve would abandon a well-posed one.

**The niche worth remembering.** BLS assumes a repeating box, so it is weakest
exactly where this model already behaves oddly: long-period and single-transit
events. 66 of the 3,919 scored candidates have a baseline covering fewer than
two periods, and 6 of the top 20 have periods over 400 days. Single- and
duo-transit detection is an area where BLS structurally underperforms and where
the community is still finding planets. If search ever enters this project,
that is the door — not a general re-run of SPOC.

### 6.2 A large language model in the pipeline

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
the structured diagnostics. For explainability, stage 11's per-branch occlusion
contributions are quantitative and faithful to what the model computed;
narrating them with an LLM would add a layer that can be wrong about our own
model, which is the opposite of the goal.

## 7. Change log

### 7.0 Incumbent became Champion — 2026-08-17

Ollie's wording: *"the best model will be known as the Champion model and weekly
refresh challenges the champion."* Code, tests and forward-facing docs now say
champion. **This document mostly does not, and that is the convention rather
than an omission.**

| surface | what happened |
|---|---|
| code, tests, `docs/index.md`, `model_pipeline.md`, `data_provenance.md`, `scripts/README.md` | renamed |
| `load_incumbent_summary` | still exported, an alias for `load_champion_summary` — the same object, so the two cannot diverge |
| `--incumbent-summary` | still accepted alongside `--champion-summary`, so commands recorded here keep running |
| `models/cv/incumbent-rebaselined/` | **unchanged on disk.** Renaming a directory is a data-layout change: it moves a DVC-tracked artefact and breaks every recorded command naming it. The vocabulary moved; the path did not |
| this document, sections 1–3 | **unchanged.** It is a record of what was measured and when, and rewriting the words a decision was taken in is not a rename |
| every pre-registration block | **unchanged, and never rewritten.** New vocabulary appears inside one only in square brackets, the convention the stage renumber used |
| `docs/handover-*.md` | unchanged; they are dated records of what a session believed |

Five forward-facing lines were renamed: the `W3` weakness, stage 8's ranking in
2g, 4.1a's statement of the defect, and 4.8's closing paragraph. Everything else
saying "incumbent" here is either a recorded result or a pre-registration, and
`incumbent` and `champion` refer to the same thing throughout: **whatever
`models/registry.json` currently serves.**

### 7.1 The 2026-08-14 record restructure

Recorded because a reshuffle of the evidence record is exactly the kind of
change that quietly loses a number.

**Verified mechanically:** every non-heading, non-blank line of the previous
`roadmap.md` is present in this one. Only heading text changed, plus the new
prose in *How to read this file*, 1d, 4.8 and this section.

| change | detail |
|---|---|
| numbered structure | flat `##`/`###` headings became `1`–`7` with `1a`/`3.2a` parts |
| section 3 reordered to chronological | the capacity-arm **launch** now precedes its **result** (the file had them inverted); *Observation baseline* moved ahead of the stage 8 pre-registration it motivates; stage 10.5's pre-registration moved out of the middle of the record into the forward plan at 4.1 |
| `plan-2026-08-09.md` deleted | its weakness register → 1d; its forward items → 4.2–4.7; its audits → 5.1–5.3; its totals rewritten at 4.8. The plan's own descriptions of hygiene, stage 7i and stage 8 were dropped as superseded by the recorded results in section 3 — they held estimates, not measurements. The file remains in git history |
| forward paragraphs pulled out of the record | the stage 9, 10, 11 and 12 paragraphs that sat inside the stage 8 result and the observation-baseline section moved to 4.2, 4.3, 4.5 and 4.7 |
| W1 status updated | the plan's "above every model" line no longer holds for arm P, which sits **below** its labels after propensity weighting |
| W13 status updated | the `npm audit fix` blocker is gone — `frontend/` is clean and the lockfile committed |
| totals re-costed | stage 10.5 added; the ~70 min per CV run figure replaced with the measured ~2 h |

**Amended 2026-08-15.** Stage 10.5's pre-registration and its amendment moved
from 4.1 into **3.11a/3.11b**, so they sit with the result whose reading they
fixed. 4.1 is now the *outstanding* part of that stage.

Reproduction for 3.11c is `ensemble.py` in the out-of-repo scratch directory.
It is **untracked**, and its own claim to have been written before the runs
finished cannot be verified — its mtime is three minutes after the last run
exited. The audit re-derived every number in 3.11c independently and they hold;
the floor it computed did not, and that is 3.11d.

**Not changed by either restructure:** every pre-registration block, every
measured number, and the stage-number mapping in 1c.

### 7.2 The 2026-08-15 documentation restructure

`docs/` was reorganised for a public reader: eight documents, no duplication,
no superseded copies. `HANDOVER.md` and five dated handovers were retired
(**W14** closed), along with the standalone architecture, features, deploy,
operating, pipeline-diagram and comparison documents, whose content was
consolidated into `getting-started.md`, `model_specs.md`, `model_pipeline.md`,
`overview.md` and `troubleshooting.md`. Measured findings and coverage
statistics are indexed in `data_provenance.md`, which is now the metrics ledger;
this file remains the record and the plan. Everything removed is in git history.
