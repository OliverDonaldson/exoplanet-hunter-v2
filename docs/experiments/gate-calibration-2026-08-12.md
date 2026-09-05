> Moved verbatim from `docs/roadmap.md` §3.7 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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
