> Moved verbatim from `docs/roadmap.md` §2f on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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
