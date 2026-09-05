> Moved verbatim from `docs/roadmap.md` §3.8 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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
