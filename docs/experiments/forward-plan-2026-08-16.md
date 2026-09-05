> Moved verbatim from `docs/roadmap.md` §? on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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

> Moved verbatim from `docs/roadmap.md` §4.3 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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

**The `difference` family must be read stratified by `dv_usable`, added
2026-08-20.** Stage 9 established that the presence gate zeroes the difference
branch wherever the DV report is absent, which is **58.9%** of this set and 100%
of Kepler and K2. An unstratified leave-one-out on that family therefore measures
a population in which the majority of rows are unaffected *by construction*: the
effect is diluted below the floor before the pass starts, and the null it returns
is uninformative rather than evidence of redundancy. The same trap cost stage 9
its primary criterion. Stratify on **stamp presence** (`n_difference_images` non-null),
not on `dv_usable` — the gate reads the view's own presence channel, and the two
differ: stamp-absent is **3,191 rows (58.8%)** while `dv_usable == False` is
**3,333 (61.4%)**. Verified 2026-08-20 against
`data/processed/viewset_scalars.parquet`. TESS itself carries 164 stamp-absent
rows (6.8%), so mission is not a proxy for it either. Report the family on
stamp-present rows, with the gated rows reported separately as the arithmetic
zero they are.

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


### 4.7 Stage 12 — UI redesign · **moved first as Phase 0 on 2026-08-20, see 4.2c**

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
