# Roadmap — the ExoMiner-inspired rebuild

Adopted 2026-07-26 after reviewing [NASA's ExoMiner](https://github.com/nasa/ExoMiner)
(ExoMiner++, TESS paper: [AJ 170, 5](https://iopscience.iop.org/article/10.3847/1538-3881/ae03a4)).

We reimplement and credit; we do not vendor their code (NASA NOSA licence).

**The index of the record.** Until 2026-09-04 this file held every measurement
and the forward plan in one place, 4,972 lines. The sections were moved
verbatim: the record to [`experiments/`](experiments/README.md), one file per
stage; the weakness register to [`known-limits.md`](known-limits.md); the
considered-and-deferred decisions to [`decisions.md`](decisions.md); the forward
plan to [`PLAN.md`](PLAN.md), which also carries the current status. Every
section number is preserved below as a pointer, so a reference such as
"roadmap 3.9b" in code, commit messages or the provenance ledger still resolves
here. Four conventions are load-bearing: **pre-registration blocks are verbatim
and never rewritten**, and a result landing outside one is reported as falsified
rather than re-specified; **`W1`–`W14`** is the weakness register; **stage
numbers were remapped once**, on 2026-08-08, with the permanent mapping at 1c;
and **nothing promotes without being asked**.
## 1. Orientation

### 1a. Why ExoMiner

Its branches target pathologies we have *measured*, not guessed:

| our measurement | what it means | ExoMiner's answer |
|---|---|---|
| corr(prob, transit count) **−0.048** | the score does not track how many transits were actually caught | unfolded per-transit branch + observed/expected transit-count scalars |
| **26.4%** of hosts pass threshold with no injection at all (46.7% planet hosts vs 12.3% FP hosts) | the model partly scores the host, not the transit | per-diagnostic branches with branch-scoped scalars, so transit evidence is explicit |
| 13-dim vetting-aux retrain: ΔAUC **−1.3e-5** | scalar summaries of diagnostics add nothing over the views | feed the odd/even, secondary and centroid **views** |
| TESS **0.906** vs Kepler **0.989** AUC | the headroom is on the mission we serve | momentum dump, transit-masked periodogram, per-sector difference-image quality |

> 2026-09-05: the 0.906 / 0.989 pair above is from the July summary this file was adopted on; the current out-of-fold figures are 0.910 / 0.991 (`models/cv/champion-rebaselined-today/cv_summary.json`).

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

*Moved verbatim to [`known-limits.md`](known-limits.md) on 2026-09-04.*

#### 1d.1 Tier 1 — defeats the product's purpose

→ [`known-limits.md`](known-limits.md)

#### 1d.2 Tier 2 — blocks delivery

→ [`known-limits.md`](known-limits.md)

#### 1d.3 Tier 3 — unexplained, and currently unowned

→ [`known-limits.md`](known-limits.md)

#### 1d.4 Tier 4 — engineering and operational risk

→ [`known-limits.md`](known-limits.md)

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
| **7ii** *(old D)* branch attribution | **deferred behind 8, 9 and 10** — 3-family sweep done 2026-08-09, **all arms null** | branch-drop mechanism built and declared in `run_config`. `unfolded`, `periodogram`, `scalar_only` all read null against the re-baseline; the one nominal PASS clears its bar by 0.23% and is an artefact of a 3-draw sd. Runs **once**, late, on a branch set and a distribution that have stopped moving. **The `difference` family must be read stratified by `dv_usable`** — 58.9% of rows have it gated off by construction, so an unstratified pass returns a diluted null (added 2026-08-20, from stage 9) |
| **8** *(old 3)* labels and negatives | **done** 2026-08-14 — four arms measured 2026-08-13, prediction 4 on 2026-08-14; **all four pre-registered predictions falsified** | **propensity weighting eliminated the architecture's amplification of the baseline confound at no measurable cost** (gap +0.1265 → −0.0071, 3.3× its bar). Synthetic negatives null; arm S unreadable by construction. The control-arm split also fell (−0.0966, 1.3× its bar) but **threshold-free host-scoring did not move**, so that second win is recorded as *qualified*. Group (a), external catalogue negatives, deliberately not done |
| **10.5** the ensemble arm | **CLOSED 2026-08-15 — BOTH ARMS CLEAR**; the control-arm pass landed the same day (3.11e) | **the branch line's value is as a complement, not a replacement.** Mean-of-logits recall @1% FPR **0.4362** (E-C) and **0.4223** (E-P) against the common-fold dual-view member's 0.3046 — **3.9x and 4.1x** their own floors. Reopens nothing about stage 4, whose rejections were about replacement. Nothing promotes |
| **9** *(old 2(d))* difference-image branch | **BUILT and measured 2026-08-20** — costs nothing, and its primary criterion **could not be measured** | the stamps were never sparse: the pixel list fills its bounding box exactly, so the re-grid is a placement not an interpolation. Branch built at 17x17 with attention over DV's per-sector quality. Recall −0.0169 (**0.20x** its floor), mission split +0.0066 (0.21x), rebuild anchor −0.0515 (0.70x) — all inside their floors. **Prediction 1, the falsification test, is unrunnable**: the stage 7i harness zeroes every DV input by its own pre-registered limit, so the branch contributes exactly 0.0 on control-arm hosts. Whether to change that limit is Ollie's call |
| **10** *(old G)* Optuna re-tune | not started | on the winner, after the distribution is settled |
| **11** *(old 4)* serving parity + explainability | not started | branch-occlusion contributions through `/score`; carries `score_std`, provenance headers, precision@k |
| **12** *(old 5)* UI redesign | **moved FIRST as Phase 0, 2026-08-20** — see 4.2c | it is the only work here decoupled from every open model question: it reads a pinned API contract that already exists, so no later stage can invalidate it, and it makes the project demonstrable at every point instead of only at the end. ExoMiner's own public vetting catalog is 270 lines of Dash with no plots (4.2b finding 7), so the product gap runs in our favour |

Two rows the old table carried separately are gone by absorption, not by
cancellation: **old 2(b)** (unfolded-flux branch, rebuilt 2026-08-08) and **old
2(c)** (trend + periodogram, built) were never separate build steps — see the
audit finding below — so what is left of both is attribution, which is stage 7.

**Serving is unchanged throughout: `ca906040` (9-dim, 2001/201) on Fly.** Nothing
in stages 1–5 has been promoted, and the registry has not been touched since
2026-07-19.

### 2b. What the 2026-08-07 audit changed about this table

*Moved verbatim to [`experiments/audit-2026-08-07.md`](experiments/audit-2026-08-07.md) on 2026-09-04.*

### 2c. Uncovered, fixed, and improved since stage 2 closed

*Moved verbatim to [`experiments/audit-2026-08-07.md`](experiments/audit-2026-08-07.md) on 2026-09-04.*

### 2d. The shared flux tower — 2026-08-07

*Moved verbatim to [`experiments/audit-2026-08-07.md`](experiments/audit-2026-08-07.md) on 2026-09-04.*

### 2e. Audit of the recorded numbers — 2026-08-07

*Moved verbatim to [`experiments/audit-2026-08-07.md`](experiments/audit-2026-08-07.md) on 2026-09-04.*

### 2f. Execution order — the dependency graph, and why it is not the numbering

*Moved verbatim to [`experiments/execution-order-2026-08-09.md`](experiments/execution-order-2026-08-09.md) on 2026-09-04.*

### 2g. What stages 7–11 are worth — ranked by impact, 2026-08-09

*Moved verbatim to [`experiments/execution-order-2026-08-09.md`](experiments/execution-order-2026-08-09.md) on 2026-09-04.*

### 2h. Is Exoplanet Hunter ready once stage 11 is done?

*Moved verbatim to [`experiments/readiness-contract-2026-08-09.md`](experiments/readiness-contract-2026-08-09.md) on 2026-09-04.*

## 3. The record — what was measured, in the order it happened

Chronological. Pre-registrations sit immediately before the result they fixed
the reading of, so the order on the page is the order the work was done in.

### 3.1 Stages 1–3 — housekeeping, ExoMiner-grade inputs, the re-baselined summary

*Moved verbatim to [`experiments/stage-01-03-inputs-and-rebaseline.md`](experiments/stage-01-03-inputs-and-rebaseline.md) on 2026-09-04.*

### 3.2 Stage 4 — per-diagnostic branches: three runs and a capacity arm

*Moved verbatim to [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md) on 2026-09-04.*

#### 3.2a Run 1 — REJECTED (2026-08-05)

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2b Pre-commitments recorded before the next result exists

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2c The trigger, re-derived against run 3 — recorded 2026-08-08, before run 3 was read

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2d K2 was unbenchmarked for 9.7% of training — now it is not

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2e Run 2 — the resolution fix, pre-registered 2026-08-06

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2f Run 2 result — the resolution hypothesis is FALSIFIED (2026-08-07)

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2g Run 3 result — the fixed architecture on the fixed shards (2026-08-08)

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2h The variance decomposition, measured for the first time

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2i The capacity arm — trigger fired, launched 2026-08-08

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2j Capacity arm result — capacity is NOT the constraint (2026-08-08)

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2k Three training-path changes that break comparability going forward

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2l The one cell three architectures have not moved

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

#### 3.2m What the run also uncovered: the noise floor was never measured

→ [`experiments/stage-04-branch-runs.md`](experiments/stage-04-branch-runs.md)

### 3.3 Stage 5 — the candidate view set, rebuilt (2026-08-08)

*Moved verbatim to [`experiments/stage-05-viewset.md`](experiments/stage-05-viewset.md) on 2026-09-04.*

### 3.4 Stage 6 — the recall noise floor

*Moved verbatim to [`experiments/stage-06-recall-floor.md`](experiments/stage-06-recall-floor.md) on 2026-09-04.*

#### 3.4a Pre-registered before the run — recorded 2026-08-08, run not yet launched

→ [`experiments/stage-06-recall-floor.md`](experiments/stage-06-recall-floor.md)

#### 3.4b First launch failed on fold 0; the pre-registration is unchanged — 2026-08-09

→ [`experiments/stage-06-recall-floor.md`](experiments/stage-06-recall-floor.md)

#### 3.4c Result — the recall noise floor, measured (2026-08-09)

→ [`experiments/stage-06-recall-floor.md`](experiments/stage-06-recall-floor.md)

### 3.5 Stage 7 — the attribution sweep, and the criterion problem

*Moved verbatim to [`experiments/stage-07-attribution-sweep.md`](experiments/stage-07-attribution-sweep.md) on 2026-09-04.*

#### 3.5a Stage 7's criterion is blocked on stage 11 — found 2026-08-09

→ [`experiments/stage-07-attribution-sweep.md`](experiments/stage-07-attribution-sweep.md)

#### 3.5b Decided 2026-08-09: the offline harness, with the incumbent re-measured on it

→ [`experiments/stage-07-attribution-sweep.md`](experiments/stage-07-attribution-sweep.md)

#### 3.5c Pre-registered before the sweep — recorded 2026-08-09, nothing launched

→ [`experiments/stage-07-attribution-sweep.md`](experiments/stage-07-attribution-sweep.md)

#### 3.5d Sweep result — every arm null, and the one "PASS" is an artefact (2026-08-09)

→ [`experiments/stage-07-attribution-sweep.md`](experiments/stage-07-attribution-sweep.md)

#### 3.5e Sequencing: should stage 8 come first? — assessed 2026-08-09

→ [`experiments/stage-07-attribution-sweep.md`](experiments/stage-07-attribution-sweep.md)

### 3.6 Stage 7i — the offline control-arm harness

*Moved verbatim to [`experiments/stage-07i-control-arm-harness.md`](experiments/stage-07i-control-arm-harness.md) on 2026-09-04.*

#### 3.6a Pre-registered before the harness runs — recorded 2026-08-09, nothing built

→ [`experiments/stage-07i-control-arm-harness.md`](experiments/stage-07i-control-arm-harness.md)

#### 3.6b Pre-registered before the measurement — recorded 2026-08-10, nothing run

→ [`experiments/stage-07i-control-arm-harness.md`](experiments/stage-07i-control-arm-harness.md)

#### 3.6c Result — the branch architecture does not score the star less (2026-08-12)

→ [`experiments/stage-07i-control-arm-harness.md`](experiments/stage-07i-control-arm-harness.md)

### 3.7 The promotion gate was not calibrated to its own noise floor (2026-08-12)

*Moved verbatim to [`experiments/gate-calibration-2026-08-12.md`](experiments/gate-calibration-2026-08-12.md) on 2026-09-04.*

### 3.8 Observation baseline — a real problem architecture cannot fix

*Moved verbatim to [`experiments/observation-baseline.md`](experiments/observation-baseline.md) on 2026-09-04.*

### 3.9 Stage 8 — labels and negatives

*Moved verbatim to [`experiments/stage-08-labels-and-negatives.md`](experiments/stage-08-labels-and-negatives.md) on 2026-09-04.*

#### 3.9a Pre-registered before stage 8 runs — recorded 2026-08-12, nothing built

→ [`experiments/stage-08-labels-and-negatives.md`](experiments/stage-08-labels-and-negatives.md)

#### 3.9b Result — the amplification is reachable, the labels are not (2026-08-13)

→ [`experiments/stage-08-labels-and-negatives.md`](experiments/stage-08-labels-and-negatives.md)

#### 3.9c Prediction 4 — the split fell, and the construct behind it did not (2026-08-14)

→ [`experiments/stage-08-labels-and-negatives.md`](experiments/stage-08-labels-and-negatives.md)

### 3.10 The UI scaffold, audited against the API — 2026-08-13

*Moved verbatim to [`experiments/ui-scaffold-audit-2026-08-13.md`](experiments/ui-scaffold-audit-2026-08-13.md) on 2026-09-04.*

#### 3.10a Which mission the console reports — decided 2026-08-13

→ [`experiments/ui-scaffold-audit-2026-08-13.md`](experiments/ui-scaffold-audit-2026-08-13.md)

#### 3.10b Two things the project already computes and the scaffold does not show

→ [`experiments/ui-scaffold-audit-2026-08-13.md`](experiments/ui-scaffold-audit-2026-08-13.md)

#### 3.10c Not required before stage 11

→ [`experiments/ui-scaffold-audit-2026-08-13.md`](experiments/ui-scaffold-audit-2026-08-13.md)

### 3.11 Stage 10.5 — the ensemble arm

*Moved verbatim to [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md) on 2026-09-04.*

#### 3.11a Pre-registered — recorded 2026-08-12, nothing run

→ [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md)

#### 3.11b Amendment — recorded 2026-08-14, before anything was built or run

→ [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md)

#### 3.11c Result — the ensemble confirms, on both arms (2026-08-15)

→ [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md)

#### 3.11d The floor's pairing was never pre-registered — the multipliers are falsified (2026-08-15)

→ [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md)

##### Result — the finding is banked, on a floor that no longer depends on a choice

→ [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md)

#### 3.11e Result — the control-arm pass, and the architecture nobody had measured (2026-08-15)

→ [`experiments/stage-10-5-ensemble.md`](experiments/stage-10-5-ensemble.md)

## 4. The forward plan — what remains, in order

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

### 4.1 Stage 10.5 — **CLOSED 2026-08-15**

*Moved verbatim to [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md) on 2026-09-04.*

### 4.1a Calibrate the refresh loop to its own noise — **CLOSED 2026-08-17**, all three defects

*Moved verbatim to [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md) on 2026-09-04.*

#### Built 2026-08-16 — defects 2 and 3 closed, defect 1 still open

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### Calibration run — 2026-08-16. It did not measure a floor; it found why there isn't one

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### The recall floor, measured — 2026-08-17. No retraining

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

### 4.1b Pre-registered — the gate's third verdict and what its floor is made of (pre-registered 2026-08-16; **implemented in `839ff8c`**)

*Moved verbatim to [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md) on 2026-09-04.*

#### 1. A third verdict — UNRESOLVED

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### 2. What the floor is made of

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### 3. The K2 alarm is permanent, and that is a decision not a bug

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

### 4.1c Pre-registered — the control lane, and what a weekly delta is allowed to mean (pre-registered 2026-08-17; **the lane is built** — its own check was FALSIFIED and is replaced by 4.1d)

*Moved verbatim to [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md) on 2026-09-04.*

#### 1. What the lane does

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### 2. Which rows decide, and why not all of them

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### 3. The two deltas, and only one of them is the model

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### 4. How the result will be read — fixed before the lane runs

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### Result — the lane is built, and its own correctness check is FALSIFIED (2026-08-17)

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

### 4.1d Pre-registered — the corrected check, which is two checks (pre-registered 2026-08-17; **run** — Check A passed bitwise)

*Moved verbatim to [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md) on 2026-09-04.*

#### Why this is a replacement and not a tolerance adjustment

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### Check A — method equivalence. Time is removed from it

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### Check B — the drift measurement. Not a gate, and not a failure

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### The staleness dependency — the one way this lane goes quietly wrong

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

#### Result — Check A passes bitwise, and the lane's first measurement is zero (2026-08-17)

→ [`experiments/refresh-gate-calibration-4-1.md`](experiments/refresh-gate-calibration-4-1.md)

### 4.2 Stage 9 — difference-image branch · 6–9 h build · 3–4 h compute

*Moved verbatim to [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md) on 2026-09-04.*

#### The blocker dissolved — the stamps were never sparse (2026-08-17)

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

#### A third state nobody had looked for — DV declines 26.6% of its own images

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

#### The control, fixed before the run

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

#### Pre-registered — recorded 2026-08-17, before either arm is launched

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

#### Built 2026-08-17 — the branch exists, neither arm has run

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

#### Amendment — recorded 2026-08-20, before either arm was launched

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

#### Result — the branch is built and measured, and its own falsification test cannot be run (2026-08-20)

→ [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md)

### 4.2b The ExoMiner++ readout — measured against the real repository and the paper (2026-08-20)

*Moved verbatim to [`experiments/stage-09-difference-image.md`](experiments/stage-09-difference-image.md) on 2026-09-04.*

### 4.2c Pre-registered — the four phases, and what each is allowed to claim (pre-registered 2026-08-20)

*Moved verbatim to [`experiments/phases-pre-registration-4-2c.md`](experiments/phases-pre-registration-4-2c.md) on 2026-09-04.*

### 4.2d Phase 1 — the target-position channel and the momentum dump

*Moved verbatim to [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md) on 2026-09-04.*

#### Built 2026-08-27 — both inputs exist, neither arm has run

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### The momentum dump — the flag is not in our light curves, and could not be

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### The epoch had to be recovered, and that is a limit worth stating

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### Three build decisions recorded, because each could have gone quietly wrong

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### Phase 1a had to run on 9b57f79's code, and finding out cost one launch

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### The floor arithmetic was verified against stage 9 before it was used

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### Two version-skew failures, and the rule they establish (2026-08-27/28)

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### The weekly refresh had never reached a verdict — fixed 2026-08-28

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

#### The promotion log — built 2026-08-28, and it was a leak rather than a feature

→ [`experiments/phase-1-build-4-2d.md`](experiments/phase-1-build-4-2d.md)

### 4.3 Stage 10 — Optuna re-tune · 2 h build · 10–13 h compute

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

### 4.4 Stage 7ii — branch attribution · 1 h build · ~7 h compute

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

### 4.5 Stage 11 — serving parity and explainability · 12–18 h build · no compute

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

### 4.6 Finishing touches · 4–6 h · no compute

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

### 4.7 Stage 12 — UI redesign · **moved first as Phase 0 on 2026-08-20, see 4.2c**

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

### 4.8 Totals, and what "finished" means

*Moved verbatim to [`experiments/forward-plan-2026-08-16.md`](experiments/forward-plan-2026-08-16.md) on 2026-09-04.*

## 5. Standing audits

*Moved verbatim to [`experiments/standing-audits.md`](experiments/standing-audits.md) on 2026-09-04.*

### 5.1 ExoMiner re-audit — not warranted, and the test applied

*Moved verbatim to [`experiments/standing-audits.md`](experiments/standing-audits.md) on 2026-09-04.*

### 5.2 Security audit — done and acted on, 2026-08-09

*Moved verbatim to [`experiments/standing-audits.md`](experiments/standing-audits.md) on 2026-09-04.*

### 5.3 Cleaning audit — done. The repo is clean; the disk is not

*Moved verbatim to [`experiments/standing-audits.md`](experiments/standing-audits.md) on 2026-09-04.*

### 5.4 The data-of-record moved mid-session, inside a docs commit — 2026-08-15

*Moved verbatim to [`experiments/standing-audits.md`](experiments/standing-audits.md) on 2026-09-04.*

## 6. Considered and deferred

*Moved verbatim to [`decisions.md`](decisions.md) on 2026-09-04.*

### 6.1 Transit search on raw light curves

*Moved verbatim to [`decisions.md`](decisions.md) on 2026-09-04.*

### 6.2 A large language model in the pipeline

*Moved verbatim to [`decisions.md`](decisions.md) on 2026-09-04.*

## 7. Change log

*Moved verbatim to [`experiments/change-log.md`](experiments/change-log.md) on 2026-09-04.*

### 7.0 Incumbent became Champion — 2026-08-17

*Moved verbatim to [`experiments/change-log.md`](experiments/change-log.md) on 2026-09-04.*

### 7.1 The 2026-08-14 record restructure

*Moved verbatim to [`experiments/change-log.md`](experiments/change-log.md) on 2026-09-04.*

### 7.2 The 2026-08-15 documentation restructure

*Moved verbatim to [`experiments/change-log.md`](experiments/change-log.md) on 2026-09-04.*
