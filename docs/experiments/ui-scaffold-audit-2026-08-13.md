> Moved verbatim from `docs/roadmap.md` §3.10 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

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
