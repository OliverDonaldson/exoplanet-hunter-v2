# Known limits

What the project has measured about its own weaknesses, and the limits every
result is read under. The register at the end is the weakness ledger moved
verbatim from `roadmap.md` §1d and frozen there. The carried limits before it
are the standing caveats later record entries added; each points at the entry
that measured it.

## Carried limits

| limit | what it means for a reading | measured in |
|---|---|---|
| **Runs over different shard sets are independent draws, not replicates** | `augment_viewset` draws stateful RNG per view, so adding a view shifts every later augmentation and dropout draw. A comparison across shard sets is one draw from an unmeasured distribution. Paired same-shard comparisons, with the drop applied at the model, are unaffected | [stage 9 result](experiments/stage-09-difference-image.md) |
| **Every floor comes from three draws** | Each seed floor in the record is estimated from three seeds and has been recorded repeatedly as too thin for the decisions it carries. The next result that leans on one should widen it first | [stage 10.5, 3.11d](experiments/stage-10-5-ensemble.md), [Phase 1a](experiments/phase-1-build-4-2d.md) |
| **A floor belongs to the architecture and run it was measured on** | Stage 6 and Phase 1a measured branch-model floors; 4.1a measured the dual-view floor (recall 0.0733). A branch floor read under dual-view numbers is a category error, and the two constants the API serves today are exactly that | [stage 6](experiments/stage-06-recall-floor.md), [4.1a](experiments/refresh-gate-calibration-4-1.md), `PLAN.md` step 4 |
| **Stage 9's primary test is unrunnable on its own instrument** | The stage 7i harness zeroes every DV input by its pre-registered limit 1, so the difference branch contributes exactly 0.0 on control-arm hosts and host-AUC cannot measure it | [stage 9 result](experiments/stage-09-difference-image.md) |
| **W7, the narrow-span high-count Kepler cell, is unexplained** | +0.1446, unmoved by two bin resolutions, four input fixes, tied odd/even weights and a shared tower. Carried as a stated limitation, not investigated; decision #33 | [stage 4, 3.2l](experiments/stage-04-branch-runs.md) |
| **There is no classical baseline** | `models/baseline_rf.py` and `conf/model/random_forest.yaml` exist with a documented rationale, but no scored cross-validated result for them exists in this repository: `mlflow.db` holds 197 runs and every one that records a model name records `cnn_dualview`. The CNN's advantage over classical ML is assumed here, not measured | `mlflow.db`, 2026-09-07; [report §4 row 1](report.md) |
| **The decision metric is an eight-to-ten row statistic** | TESS recall @1% FPR is cut at the 10th-highest negative score (8th on `dv_usable`), so its paired sampling sd is **0.0437** — 2·sd ≈ 0.087, the same order as the seed floors the record reads against. Three members of one arm span 0.1418–0.2221 on identical data. Any architecture effect below ~0.09 on this metric is undetectable at this sample size, however many models are trained | paired bootstrap on `models/phase1/arm-{c,d}-*/predictions.parquet`, 2026-09-07 |
| **Three view-set artefacts drift from their DVC pointers** | `viewset.npz`, `viewset_tfrecords` and `viewset_scalars.parquet` on disk differ from what `dvc pull` returns; stage 10.5 trained on the on-disk bytes. Open decision #31 | `dvc status`, 2026-09-04; issue #31 |

## The weakness register (frozen)

> Moved verbatim from `docs/roadmap.md` §1d on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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
| **W1** | **Ranking is driven by observation baseline, and the signal is in the labels** | corr(baseline, label) **+0.387 TESS** — TESS confirmed planets median 1,495 d vs 430 d for FPs. **The pooled "+0.278 all" is stale and was withdrawn 2026-08-20**: re-measured on `viewset_scalars.parquet` it is **+0.2136**, and pooling hides opposite signs — Kepler **+0.1025**, K2 **−0.1490**. **W1 is a TESS defect, not a global one**, and every pooled reading of it in this file is to be read against that | **stage 8 — done 2026-08-14, half delivered** |
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
| **W6** | **`score_std` is computed and thrown away** | not persisted, not in the catalogue, not surfaced per candidate — and it is a real differentiator (ExoMiner concedes theirs "is NOT a probability" — their code, `vetting_tce_catalog_exominer_dash_app.py:81`, not the paper) | **finishing touches** |

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

### Notes appended to the frozen register

> 2026-09-05. "This file" in the register's preamble meant `roadmap.md`, where
> the block was written; the handovers it mentions were retired the same day.
> The W labels are now referenced from the record files and the plan.

> 2026-09-05. **W6 is stale.** `prob_std` is persisted in
> `results/candidates_scored.parquet`, joined onto the catalogue and exposed
> on `CandidateRow`; the vetting page shows the MC-dropout band per score.
> What remains of W6 is surfacing the fold and MC spread columns (step 6).

> 2026-09-05. The five rejected arms in W3, in order: run 1
> `branches-20260805` 0.238; run 2 `branches-20260807-2001` 0.126; run 3
> `branches-20260807-shared` 0.145; the capacity arm
> `branches-20260808-capacity` 0.236; the re-baseline
> `branches-20260808-rebaseline` 0.220. All against the champion's 0.307.

> 2026-09-05. **W10 measured inside a single run.** The register records the
> slowdown across repeated `run_cv` calls in one process. Phase 1's arm D''
> shows the same pathology within one call: fold completion times of 25, 23,
> 29, 36 and 78 minutes, the last fold 3.1x the first on identical work. That
> accounts for most of the 74-minute gap between the two arms rather than the
> one extra branch. The mitigation is unchanged and it held — one process per
> CV run. See [phase 1 result](experiments/phase-1-result.md).
