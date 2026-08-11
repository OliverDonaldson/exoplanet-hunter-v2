# Handover — stage 8, labels and negatives

**You are starting a fresh session to run stage 8.** Read this first, in full,
then `docs/plan-2026-08-09.md`, then `docs/roadmap.md`. Do not start from
`HANDOVER.md` — it is superseded and every stage label in it is an old one.

---

## 0. Before anything: confirm your objective, then check the gate

**First, say back what you are here to do and what you plan to do about it.**
Ollie has asked that every new session state its objective and lay out its plan
*before* touching anything. One paragraph of intent, then the checklist below,
then a plan. Not code.

**Then run this gate. Stage 8 must not begin until stage 7i is closed and
committed.** Stage 8 changes the training distribution, which invalidates
stage 7's numbers — so if 7i is unfinished, starting here destroys work that has
already been paid for.

```bash
cd /Users/ollie/Project/v2 && \
  ls results/control_arm/branches-20260808-rebaseline.json \
     results/control_arm/ca906040cdb74ba6b07353a500244777.json && \
  grep -c "Stage 7i result" docs/roadmap.md && \
  git status --porcelain --untracked-files=no && \
  git log --oneline -3
```

**All four must hold:**

| check | pass condition |
|---|---|
| both result JSONs exist | two files listed, neither empty |
| the result is written up | `grep -c` returns **≥ 1** |
| no tracked file is dirty | `git status --porcelain -uno` prints **nothing** |
| the write-up is committed | a `docs(stage 7i)` or `feat(eval)` commit at HEAD |

**If any check fails, stop and tell Ollie.** Do not "finish 7i quickly first" —
report what is missing and wait. Two known-untracked files are expected and are
*not* yours to commit: `docs/demo-script-2026-08-08.md`, and an `animejs`
addition in `frontend/package.json` + `package-lock.json`.

---

## 1. The non-negotiables

Stated in full in `docs/handover-2026-08-08.md`; they have held all the way
through and they are not style preferences.

1. **Environment.** `source /opt/anaconda3/etc/profile.d/conda.sh && conda
   activate exoplanet-hunter-v2` before anything. The V1 env shadows the package
   with V1 code and has silently run the wrong trainer before.
2. **Never promote.** `ca906040` serves. Do not touch `models/registry.json`.
3. **Guards raise.** This project's defining failure mode is a check that returns
   a plausible answer instead of failing. No broad `try/except`. A guard that
   cannot be observed to fire is not a guard — make it fire in a test.
4. **One test file per source module.** No casual scripts; scratch work goes to
   the scratchpad, not the repo.
5. **Pre-registration is binding.** Write down how a result will be read
   *before* the run finishes, in `roadmap.md`. Nobody is watching an autonomous
   session read its own result. If a result lands outside its pre-registration,
   report it as falsified — do not re-specify the criterion.

**Stop and ask** for: anything touching `models/registry.json` or Fly; a result
outside its pre-registration; re-scoping the stage; deleting non-re-derivable
data or anything with a DVC pointer; `git push` / `dvc push` — commit freely,
pushing is Ollie's.

---

## 2. What stage 8 is, and why it is the most valuable thing left

**The defect.** Observation baseline correlates **+0.278 with the ground-truth
label**, and **+0.387 on TESS alone** — *above every model*. TESS confirmed
planets have a median baseline of **1,495 days against 430** for false
positives. The mechanism is confirmation bias in the catalogue: a target
observed across many sectors accumulates the follow-up that promotes it to
confirmed, while a briefly-observed one stays a candidate or is retired.

**Why no architecture can fix it.** The model learned it because *in the
training labels it is true*. Every model sits below the labels on this
correlation. This is the only stage that can reach it.

**Why it matters for the product.** The deployment use is ranking candidates for
follow-up. Baseline dependence actively defeats that purpose: it promotes
targets that already received attention over under-observed ones that may
deserve it.

Stage 8 is ranked **#1 by impact** in `plan-2026-08-09.md`, and it is also the
**lowest-confidence** item on the plan.

---

## 3. Contents

Three groups. The third is the one that carries the stage.

**a. Negatives from external catalogues.** EB-catalogue and brown-dwarf
negatives, plus the ephemeris-match test.

**b. Synthetic negatives** — scrambled and inverted light curves, built with the
existing injection machinery (`eval/injection_recovery.py`, already tested).

**c. The three interventions against baseline dependence:**
- propensity-score weighting on observation baseline,
- baseline-stratified negative sampling,
- synthetic negatives that break the correlation by construction.

---

## 4. The kill criterion — read this before you start ingesting

**If external catalogue ingestion exceeds ~8 hours without a usable negative
set, stop and fall back to the synthetic negatives alone.**

They need no external catalogue, they break the correlation by construction, and
they have the cleanest causal story of the three interventions. This is recorded
in `plan-2026-08-09.md` and it is a decision already made — you are not
re-litigating it, you are executing it.

The estimate is soft because the only precedent for external ingestion here is
the DV download: sized at 14–56 GB, actual 3.6 GB in 5.3 h. The estimate was 5×
out — in the safe direction, but out.

---

## 5. You already have the measuring instrument. Use it.

**Stage 7i built the offline control-arm harness, and it is stage 8's
instrument, not only stage 7's.** This is why 7i was sequenced first.

- `pipeline/src/exoplanet_hunter/eval/control_arm.py` — baseline-matched host
  selection, both operating points, the pass-rate report. 15 tests.
- `pipeline/scripts/control_arm.py` — the driver. Two lanes, selected from the
  checkpoints on disk: eleven-view branch runs and the dual-view incumbent.

The clean test of stage 8 was pre-registered on 2026-08-06 as pre-commitment
(d): **injection-recovery on matched hosts with observation baseline held
constant.** That is this harness. Run it before and after your interventions.

**The before-reading exists** — that is what stage 7i produced. Do not re-derive
it; compare against it, and note that a distribution change means the branch
model must be retrained before its control-arm number means anything new.

Useful flags: `--per-stratum 200` takes the pool's maximum (580 hosts, 290 per
label), `--also-routable-in <other run>` restricts to hosts routable in both,
and `--shard-dir` persists the built shard set so a failed scoring pass does not
cost the ~25 min build again.

---

## 6. Two knock-ons to budget for

Both are stated in the roadmap and both are easy to forget:

1. **Changing the label distribution invalidates the re-baselined incumbent
   summary** from stage 3. It needs regenerating — one command
   (`evaluate.py summarise`), and a reason to keep stage 3 a repeatable path
   rather than a one-off artefact.
2. **It invalidates stage 7's attribution numbers.** That is exactly why stage 8
   sits ahead of stage 9, and why stage 7ii runs late.

---

## 7. Operational facts that will cost you hours

- **A CV run is ~70 min** at 5 folds × 3 models per fold.
- **Run the test suite one process per file.** Repeated `run_cv` calls in a
  single process get monotonically slower — 86s, 108s, 161s, 190s for four
  *identical* runs, and a file with eighteen of them did not finish in three
  hours. `clear_session()` does **not** help; the cause is open.
  ```bash
  for f in pipeline/tests/test_*.py; do python -m pytest "$f" -q -p no:randomly; done
  ```
  Whole suite ~5.5 min this way.
- **Never pipe a long run through `tail`** — it buffers, so a three-hour run
  gives zero progress. Redirect to a log and tail the log.
- **`git add docs/` is a trap.** It has swept another session's untracked file
  into a commit. Stage explicit paths, always.
- **The index is not the working tree.** A `git mv` staged earlier in a session
  lands in the *next* commit even if you only `git add` two other paths. Check
  `git diff --cached --name-status` before committing.
- **An eager `GradientTape` over the assembled branch model aborts the
  process** — `Fatal Python error: Aborted` on TF 2.17.1 / Keras 3.15.0 on
  Metal. `model.fit` is unaffected. Probe an isolated layer instead.
- **Two TESS populations coexist.** Headline figures (0.9130 / 0.145 / 0.307)
  are the n=2,367 shared-join gate slice; a run's own `per_mission.TESS` block
  is n=2,399 and reads differently. Both are correct — state which you quote.

---

## 8. State at handover

Branch `v2`, a worktree of `/Users/ollie/Project`. Pushed to `v2origin/main`
through `a590290`; everything after is **local and unpushed — pushing is
Ollie's**, as is `dvc push`.

Nothing has been promoted. `models/registry.json` untouched; `ca906040` served
since 2026-07-19.

Tests: **44 api**, full pipeline suite green one-process-per-file. ruff,
ruff-format and mypy clean.

---

## 9. What to produce

1. A pre-registration in `roadmap.md`, written **before** any run finishes:
   what will be measured, against what, and how each outcome reads.
2. The interventions, with tests.
3. The residual baseline correlation, **quantified** against the +0.278 / +0.387
   it started at — the deliverable is the number, not the intervention.
4. A regenerated stage-3 incumbent summary.
5. A dated write-up in `roadmap.md`, and a handover for whoever comes next.

**The honest framing to keep in view:** the project's value does not depend on
the branch architecture winning. Five arms have been rejected. If the branch
line never beats `ca906040`, the delivered outcome is still a calibrated,
explainable, reproducible vetting service, plus the label-bias work in this
stage — which improves *any* model, including the one that is actually served.
Only "we never found out" is a failure.
