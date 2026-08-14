# Roadmap — the ExoMiner-inspired rebuild

Adopted 2026-07-26 after reviewing [NASA's ExoMiner](https://github.com/nasa/ExoMiner)
(ExoMiner++, TESS paper: [AJ 170, 5](https://iopscience.iop.org/article/10.3847/1538-3881/ae03a4)).
The UI redesign stays the locked final step.

We reimplement and credit; we do not vendor their code (NASA NOSA licence).

## How to read this file

**This is the single record of the project: what was measured, and what is
left.** It absorbed `plan-2026-08-09.md` on 2026-08-14 — that document had
gone stale (it predated stage 10.5 entirely, and its cost model was out by
1.7x), and two documents disagreeing about what comes next is worse than one
that is occasionally wrong. The deleted plan remains in git history.

**Numbering.** Sections are `1, 2, 3`; their parts are `1a, 1b` or `3.1, 3.2`;
sub-parts are `3.2a, 3.2b`. Section 3 is ordered by **when the work happened**,
so it reads as provenance: each result sits after the pre-registration that
fixed how it would be read.

**Four conventions that are load-bearing.**

1. **Pre-registration blocks are verbatim and are never rewritten.** They are
   headed *Pre-registered before …* and carry the date nothing had been run.
   A result landing outside its pre-registration is reported as falsified, not
   re-specified. Their stage numbers are the ones current when they were
   written; new numbers appear only in square brackets.
2. **`W1`–`W14` are the weakness register** — see 1d. `W` is just *weakness*,
   numbered and ranked by damage to the product's actual job.
3. **Stage numbers were remapped once**, on 2026-08-08. The permanent old→new
   table is 1c; run directories and commit messages still carry old labels.
4. **Nothing promotes without being asked.** `ca906040` has served since
   2026-07-19.

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
| **W3** | **No branch model has ever beaten the incumbent where it is used** | five arms rejected, all on shortlist recall: 0.238 / 0.126 / 0.145 / 0.236 / 0.220 against **0.307** | **stage 10.5** tests *complement*, then **stage 10**, then a written decision |

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
| **W14** | **`HANDOVER.md` is 2,077 lines, superseded and partly wrong** | carries old stage labels throughout; three documents point around it | **low — it opens with a SUPERSEDED banner.** Retiring it is a *finishing* task: it still holds the only copy of the stage-2 sizing measurements and the merge collision that dropped the transit counts, and `roadmap.md` points at those rather than duplicating them |

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

One table, kept current. Detail for each row is in the stage sections below and
in HANDOVER.md.

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
| **1** | **8** labels and negatives | defect 5 | **The largest measured defect, and the only stage that can reach it.** Baseline correlates +0.278 with the label itself and +0.387 on TESS — *above every model*, so no architecture can touch it. For the deployment use it is actively counterproductive: it promotes targets that already received attention over under-observed ones that may deserve it. It improves **any** model, including the served incumbent | 25–35 h, **low** — external catalogue ingestion, whose only precedent was 5× out |
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

Against the seven-point "what finished means" contract in
`handover-2026-08-08.md` — **yes, with two named exceptions.**

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
past all seven gates, in HANDOVER.md (2026-08-05).

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

Reproduction: `~/Downloads/.stage8-scratch/prediction4.py` rebuilds every figure
in this subsection from the two `results/control_arm/stage8-*.parquet` files.

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
nothing else, and neither added row is in the incumbent's scored set. **And had
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

### 4.1 Stage 10.5 — the ensemble arm · next · build + ~8 h compute

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


### 4.5 Stage 11 — serving parity and explainability · 10–15 h build · no compute

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
| retire `HANDOVER.md` — archive it and update every reference | **W14** |
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

**Re-costed 2026-08-14.** The plan's original table is superseded: it omitted
stage 10.5 entirely (pre-registered three days after the plan was written) and
inherited a "~70 min per CV run" figure that stage 8 measured at ~2 h.

| order | item | build | compute | total |
|---|---|---:|---:|---:|
| 1 | stage 10.5 ensemble arm | 8–10 h | ~8 h | **16–18 h** |
| 2 | stage 9 difference-image branch | 6–9 h | 3–4 h | **9–13 h** |
| 3 | stage 10 Optuna re-tune | 2 h | 10–13 h | **12–15 h** |
| 4 | stage 7ii branch attribution | 1 h | ~7 h | **8 h** |
| 5 | stage 11 serving parity | 10–15 h | — | **10–15 h** |
| 6 | finishing touches | 4–6 h | — | **4–6 h** |
| 7 | stage 12 UI redesign | *unestimated* | — | *unestimated* |
| | | | | **~59–75 h** plus the UI |

**Stage 10.5 carries two build items no earlier plan costed**, both found
2026-08-14 and both reusable rather than one-offs: neither trainer can accept
an external fold assignment, and the dual-view trainer has no
`n_models_per_fold`. Stages 9 and 7ii face the same cross-run comparability
problem the fold artefact solves.

**One dependency worth stating.** If stage 10.5 clears its bar, the branch line
survives as a **complement** rather than a replacement, and stage 11 then has
to serve two models rather than one. Its 10–15 h assumes one. Partly mitigated
by `ScoringEnsemble.from_registry` already loading `cnn_dualview.keras`, but it
is not costed above.

**Finished** is the seven-point contract in `handover-2026-08-08.md`. After the
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
incumbent, and an evaluation apparatus that can tell a real improvement from
noise. Only "we never found out" would be a failure.

## 5. Standing audits

Done, acted on, and kept here so they are not re-run from scratch. Merged from
`plan-2026-08-09.md`; the dates are when each was performed.

### 5.1 ExoMiner re-audit — not warranted, and the test applied

`docs/exominer-comparison-2026-08-07.md` is a 42 KB deep dive: 10 ranked
adoptions, six questions answered against their source, what v2 already does
better, and an explicit "do not copy" list. A re-read would re-derive it.

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

## 7. Change log — the 2026-08-14 restructure

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

**Handovers dated before 2026-08-14 still reference the deleted plan** —
`handover-2026-08-08.md`, `handover-stage-8.md` and `handover-stage-8-close.md`.
They are historical records and the project does not edit them, so the references
are left dangling on purpose; `docs/index.md` says where to read instead.
`handover-2026-08-14.md` is the live one and was updated.

**Not changed:** every pre-registration block, every measured number, and the
stage-number mapping in 1c.
