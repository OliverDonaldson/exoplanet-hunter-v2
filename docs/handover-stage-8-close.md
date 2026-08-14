# Handover — closing stage 8

**You are starting a fresh session to finish stage 8.** Read this in full, then
the *Stage 8 result* section in `docs/roadmap.md`, then `docs/plan-2026-08-09.md`.
Do not start from `HANDOVER.md` — it is superseded.

**Stage 8's four training arms are DONE and written up. One measurement is
outstanding.** Most of this stage is already banked; do not re-run it.

---

## 0. Before anything: confirm your objective, then check the gate

**Say back what you are here to do and lay out your plan before touching
anything.** One paragraph of intent, then the checklist, then a plan. Not code.

```bash
cd /Users/ollie/Project/v2 && \
  ls models/cv/stage8-{control,propensity,stratified,synthetic}/cv_summary.json && \
  grep -c "Stage 8 result" docs/roadmap.md && \
  git status --porcelain --untracked-files=no && \
  git log --oneline -3
```

| check | pass condition |
|---|---|
| four arm summaries exist | four files listed |
| the result is written up | `grep -c` returns **≥ 1** |
| no tracked file is dirty | prints **nothing** |
| HEAD is a real commit | any |

**Two untracked directories are expected and are NOT yours to commit:**
`data/processed/synthetic_negatives/` and `data/processed/viewset_tfrecords_synneg/`.
They are DVC-territory data. Their presence means `git_dirty: True` is recorded
by every run — expected, consistent across all four arms, not a red flag.

---

## 1. The non-negotiables

Unchanged from `docs/handover-stage-8.md` §1. In brief:

1. **Environment.** `source /opt/anaconda3/etc/profile.d/conda.sh && conda
   activate exoplanet-hunter-v2` before anything.
2. **Never promote.** `ca906040` serves. Do not touch `models/registry.json`.
3. **Guards raise.** A guard that cannot be observed to fire is not a guard.
4. **One test file per source module.** Scratch work goes outside the repo.
5. **Pre-registration is binding.** A result outside its pre-registration is
   reported as falsified, never re-specified.

**Stop and ask** for: `models/registry.json` or Fly; a result outside its
pre-registration; re-scoping; deleting non-re-derivable data; `git push` /
`dvc push` — commit freely, pushing is Ollie's.

---

## 2. What stage 8 found

Full detail in `roadmap.md` under **Stage 8 result**. The headline:

**Propensity weighting eliminated the branch architecture's amplification of the
observation-baseline confound, at no measurable cost.** On the frozen 2,399-row
out-of-fold TESS slice, the amplification gap (`score↔baseline` minus
`label↔baseline`) went **+0.1265 → −0.0071**, which is **3.3× its own measured
bar**, while TESS AUC and shortlist recall both stayed inside their noise floors.

Synthetic negatives were **null** (0.1× bar). Stratified sampling is **not
comparable** — it never scored 680 of the pre-registered rows.

**Three of four pre-registered predictions were falsified.** The fourth is what
you are here to measure.

---

## 3. What is outstanding

### 3a. Prediction 4 — the control-arm split *(the only measurement left)*

Run the stage 7i harness on **both** `stage8-control` and `stage8-propensity`,
paired on the same 580 hosts. Both arms' fold assignments are byte-identical, so
the draw comes out the same — verified.

```bash
cd /Users/ollie/Project/v2
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate exoplanet-hunter-v2
INC=models/cv/ca906040cdb74ba6b07353a500244777
for arm in control propensity; do
  python pipeline/scripts/control_arm.py \
    --run-dir models/cv/stage8-$arm \
    --also-routable-in $INC \
    --per-stratum 200 --seed 42 \
    --shard-dir ~/Downloads/.stage8-scratch/ca-shards-$arm \
    --out results/control_arm > ~/Downloads/.stage8-scratch/ca-$arm.log 2>&1
done
```

**~50 min build + ~1 min scoring per arm, so ~1 h 45 for both.** Confirm the log
says `580 hosts (290 planet / 290 FP)` and `1051 routable in both runs` — that
matches stage 7i and means the draw is the pre-registered one.

**How it reads** (pre-registered): a model that vets the transit should show a
**smaller planet-minus-FP split**. The before-reading is **+0.1195**
(`branches-20260808-rebaseline`, stage 7i). Prediction 4 says it does *not* move.

- **Split falls beyond the bar** → propensity weighting reduced host-scoring as
  well as amplification. A second, independent win.
- **Split level** → prediction 4 confirmed. Baseline dependence and host-scoring
  are different defects and fixing one leaves the other, which is what stage 7i
  already implied.

Run the control arm too, not just propensity — stage 8's control is a fresh
retrain and its split could differ from +0.1195 through reseeding alone.

### 3b. Regenerate the stage-3 incumbent summary

`evaluate.py summarise`. The label change invalidates it. One command.

### 3c. Finish the write-up and hand over

Add prediction 4's outcome to the *Stage 8 result* section, then write the next
handover.

---

## 4. Open decisions for Ollie — ask, do not assume

1. **External catalogue negatives (group a) were never started.** EB and
   brown-dwarf catalogues plus the ephemeris-match test, with an 8 h kill
   criterion. Arm P has already delivered the stage's reachable deliverable, so
   the case for spending up to 8 h more is weak. **Recommend skipping and
   recording it as deliberately not done.** Ollie's call.
2. **Stage 10.5, the ensemble arm**, is pre-registered in `roadmap.md` and runs
   after stage 8. It is the largest measured effect in the project
   (mean-of-logits recall 0.4746 vs the incumbent's 0.3069, 5.7× the floor) and
   is explicitly **not a result** until a common-fold confirmation run exists.
3. **Arm S is unreadable.** Re-running it with the dropped rows restored to the
   *test* split would make it comparable. Worth ~2 h? Probably not, given P
   already answered the question.

---

## 5. Operational facts that cost this session hours

- **Nothing long-running goes in `/private/tmp`.** It cost us twice: once
  clearing helper scripts mid-session, once losing every log and shard dir to a
  reboot. Use `~/Downloads/.stage8-scratch/`.
- **`setsid` does not exist on macOS.** Use `screen -dmS <name> caffeinate -dimsu
  <script>` — that reparents to PID 1 and survives the session.
- **Do not hold a long Bash waiter against a detached job.** A block died when
  the waiter was terminated and took the shared process group with it. Poll for
  the artefact instead (`cv_summary.json` appearing).
- **`caffeinate`'s `PreventSystemSleep` only works on AC power.** On battery it
  blocks idle sleep only, and a drained battery sleeps regardless. Check
  `pmset -g assertions` shows `PreventSystemSleep 1`, not just the idle one.
- **A CV run is ~2 h** at 5 folds × 3 models, not the ~70 min the 2026-08-08
  handover quotes.
- **Run the test suite one process per file**; `test_train_branches.py` alone is
  ~6 min.
- **Never edit `pipeline/src/**` or `pipeline/scripts/train_branches.py` while a
  multi-arm block runs** — the runner launches a fresh interpreter per arm, so
  the arms would train on different code. `api/`, `frontend/`, `docs/` and
  `pipeline/tests/` are safe; verified `control_arm.py` cannot reach
  `train_branches` transitively.
- **Every commit changes the `git_sha` recorded by runs launched after it.** The
  four stage-8 arms span two SHAs; `git diff 5da64de 51e256b -- pipeline/` is
  empty, so the delta is docs-only. Batch doc commits until a block finishes.

---

## 6. State at handover

Branch `v2`, a worktree of `/Users/ollie/Project`. Nothing promoted;
`models/registry.json` untouched; `ca906040` served since 2026-07-19. Nothing
pushed — pushing is Ollie's.

Also landed this session, outside stage 8 proper:

- **The promotion gate was audited and fixed** (`2dc3dd4`). It could not promote
  anything — it refused every candidate on paperwork before comparing a metric.
  Its recall tolerance was 0.02 against a measured floor of 0.0337. A tie on AUC
  read as a defeat. All three fixed; every recorded rejection still stands.
- **Stage 10.5 pre-registered** (`c9266cb`).
- **The UI scaffold audited against the API** (`51e256b`) — including that the
  design had nowhere to put stage 11's per-branch explainability, now resolved
  with a fourth tab.
- **Per-epoch training curves persisted** (`81c7a89`). The four stage-8 arms
  predate this and carry none; no number moves, so they stay comparable.

Reproduction recipe for the stage-8 table: `~/Downloads/.stage8-scratch/measure.py`
rebuilds it from the four `predictions.parquet` files.
