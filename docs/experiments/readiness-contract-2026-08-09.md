> Moved verbatim from `docs/roadmap.md` §2h on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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
