> Moved verbatim from `docs/roadmap.md` §3.9 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.9 Stage 8 — labels and negatives

**Why this stage exists, and why it was moved.** Recorded 2026-08-08,
before any of it ran.

**Stage 8 *(old 3)* — labels and negatives.** EB-catalogue and brown-dwarf
negatives, the ephemeris-match test, and scrambled/inverted synthetic negatives
built with our existing injection machinery. Plus the observation-selection
problem below, which arrived here from the branch-model work.

**Moved ahead of stage 9 on 2026-08-08.** Two reasons. Stage 8's
interventions change the training distribution, so anything measured before it
has to be re-measured after — and stage 9 is the expensive one, so the roadmap
order paid for it twice. And on the evidence, two architecture runs have now
been rejected while the largest measured defect sits at **+0.278 in the labels
themselves, +0.387 on TESS** — above every model, and somewhere no architecture
can reach. The original ordering was set before either of those facts existed.

Two knock-ons to expect: changing the label distribution invalidates the
re-baselined incumbent summary from stage 3, which will need regenerating — one
command, and a reason to keep stage 3 a repeatable path rather than a one-off
artefact — and it invalidates stage 7's attribution numbers, which is exactly why
stage 8 sits ahead of stage 9.



#### 3.9a Pre-registered before stage 8 runs — recorded 2026-08-12, nothing built

Written before any intervention exists, because nobody is watching an autonomous
session read its own result.

**The before-readings already exist and are NOT to be re-derived.** Stage 7i
produced the control-arm numbers; the baseline sensitivities were re-measured
per mission on 2026-08-12. Everything below is compared against these:

| statistic, TESS gate slice | before |
|---|---:|
| Spearman(branch score, `baseline_days`) | **+0.5155** |
| Spearman(label, `baseline_days`) | **+0.3874** |
| **amplification gap** = score − label | **+0.1281** |
| control-arm split, branch, F1 cut | **+0.1195** |
| TESS AUC / recall @1% FPR | 0.9202 / 0.2196 |

**Two targets, separated. This is the point of the decomposition.** Stage 8's
framing until now was "the bias is in the labels, no architecture can reach it".
That is half the story:

- **Target A — the labels.** `Spearman(label, baseline_days)`. The interventions
  change what the training labels *are*, so this moves by construction and its
  movement is the intervention working as designed.
- **Target B — the amplification.** `score − label`. The branch model sits
  **+0.1281 above** its own labels on TESS. No label intervention is required to
  move this, and an intervention that fixes A while leaving B is a partial
  result. Reporting only the score correlation would conflate the two, which is
  how the pooled figure hid the crossing for four days.

**The evaluation population is frozen, and this is the trap to avoid.** Adding
negatives changes the training population, so a correlation computed over the
*augmented* set is not comparable to +0.5155. Every number above and after is
measured on the **same out-of-fold TESS rows** as the before-reading. Synthetic
and external negatives enter **training only** and are excluded from every
evaluation slice. A run that cannot demonstrate that exclusion is not read.

**The bar.** Sampling error on a Spearman ρ at n=2,399 is `1/√(n−3) ≈ 0.0204`,
so **2σ ≈ 0.041** — but that is a fixed model's sampling error, and reseeding
noise on this statistic has never been measured here. So each run reports the
**per-member spread** of its own baseline sensitivity at `--n-models-per-fold 3`,
and the bar is stage 6's rule — `2 × sd / √3` — floored at the 0.041 sampling
bar. **A margin under the larger of the two is not a decision.**

**Arms, and why they are separate runs.** Three interventions run together
cannot be attributed, and this project has already paid for that once with the
three-family sweep. Each is measured against the same control:

| arm | what changes |
|---|---|
| control | the current distribution, re-run for a same-code comparison |
| **P** propensity weighting | per-example weights ∝ inverse propensity on `baseline_days` |
| **S** stratified negatives | negatives resampled to match the positives' baseline distribution |
| **N** synthetic negatives | scrambled + inverted light curves, baseline-independent by construction |
| combined | only if at least one single arm clears its bar |

At ~70 min a run that is ~4.7 h for the four, inside the 6–8 h budgeted.

**How each outcome reads — fixed now.**

| outcome | reading |
|---|---|
| amplification gap **falls** beyond the bar | the architecture's contribution to baseline dependence is reachable. Report the arm and the residual |
| gap **level**, label correlation falls beyond the bar | the label intervention worked and the architecture still amplifies. **A partial result, reported as partial** — this is the outcome the old framing would have called a success |
| **neither** moves beyond its bar | the intervention is falsified. Record it; do not re-specify, and do not run a fourth arm looking for a better one |
| baseline dependence falls but **TESS recall @1% FPR falls beyond its 0.0337 floor** | the fix costs shortlist performance. Report both numbers together; a bias fix that defeats the deployment use is not a fix |

**Predictions, recorded so they can be wrong.**

1. **N (synthetic) moves the amplification gap the most**, because it is the only
   arm that breaks the correlation by construction rather than reweighting an
   existing population.
2. **P (propensity weighting) moves the label correlation but not the gap** —
   reweighting changes what the model is fitted to, not how it extrapolates.
3. **At least one arm costs shortlist recall beyond its floor.** Observation
   baseline is genuinely predictive of the label in this catalogue; removing the
   model's access to it should cost measured performance, and an intervention
   that costs nothing would be evidence it did nothing.
4. The control arm's split (+0.1195) **does not move** on any arm, because
   host-scoring and baseline dependence are different defects — stage 7i already
   showed the split survives baseline matching.

**The kill criterion stands as a decision already made.** If external catalogue
ingestion exceeds **~8 hours** without a usable negative set, fall back to the
synthetic negatives alone. Not re-litigated here — executed.

**Nothing promotes.** Whatever stage 8 measures, `models/registry.json` is
untouched and `ca906040` stays served.

#### 3.9b Result — the amplification is reachable, the labels are not (2026-08-13)

Four arms, `--n-models-per-fold 3`, ~2 h each. **Prediction 4 was measured on
2026-08-14** through the stage 7i harness and is written up below; the stage is
now complete.

**The evaluation population is the full out-of-fold TESS slice the before-reading
was taken on, n=2,399, identical rows for every arm.** Recorded because the first
attempt at this table intersected the four arms' rows instead, and that is wrong
in a way worth naming: **the intersection *is* the stratified arm's kept rows**,
a population that arm engineered to be free of the confound. Its own label
correlation there is **+0.0573 against the slice's +0.3874**, so comparing arms on
it would have refereed the contest with one contestant's own instrument.

| arm | TESS AUC | recall @1% FPR | score↔baseline | label↔baseline | **gap** |
|---|---:|---:|---:|---:|---:|
| control | 0.9204 | 0.2506 | +0.5139 | +0.3874 | **+0.1265** |
| **P propensity** | 0.9138 | 0.2642 | **+0.3803** | +0.3874 | **−0.0071** |
| N synthetic | 0.9127 | 0.2460 | +0.5097 | +0.3874 | +0.1223 |

**The control reproduces the before-reading**: score +0.5139 against +0.5155, gap
+0.1265 against +0.1281. An independent retrain landing on the same numbers is
what makes the rest of the table readable.

| arm | Δ gap | vs bar | Δ AUC | vs floor | Δ recall | vs floor |
|---|---:|---|---:|---|---:|---|
| **propensity** | **−0.1336** | **3.3×** | −0.0066 | 0.8× *(level)* | +0.0136 | 0.3× *(level)* |
| synthetic | −0.0042 | 0.1× *(null)* | −0.0076 | 0.8× *(level)* | −0.0045 | 0.1× *(null)* |

Bars are each run's own per-member spread by stage 6's rule, floored at the
0.0409 Fisher sampling bar, exactly as pre-registered.

**Propensity weighting eliminated the architecture's amplification of the
confound at no measurable cost.** The branch model went from sitting **+0.13
above** its own labels to fractionally below them: it no longer amplifies, it
merely inherits. That is the pre-registered outcome *"the architecture's
contribution to baseline dependence is reachable"*.

**What it does not claim.** Target A — the bias in the labels — is untouched, and
cannot be otherwise: the label correlation on a frozen evaluation slice is
+0.3874 by definition. What is gone is target B, the amplification. Stage 8's
deliverable is therefore **half of what the stage set out to reach, and it is the
half no architecture change could have delivered.**

**Arm S is not comparable, and that is structural rather than a failure.** It
never scored 680 of the 2,399 pre-registered rows, because rows dropped before
the split are absent from training, validation *and* test. Its own-slice figures
(AUC 0.8799, recall 0.0777) are measured on a rebalanced population where the 1%
FPR threshold means something else, and they are not evidence about the
intervention. The build-time note that excluding rows from test was "the honest
reading" was correct and incomplete: it also makes the arm unreadable against a
fixed evaluation slice. **A resampling intervention has to keep the evaluation
population whole even when it changes the training one.**

**All four predictions falsified.**

| # | prediction | outcome |
|---|---|---|
| 1 | N moves the gap most | **falsified** — N moved it least (0.1×), P most (3.3×) |
| 2 | P moves the label correlation but not the gap | **falsified**, exactly backwards |
| 3 | at least one arm costs shortlist recall beyond its floor | **falsified** — neither comparable arm did |
| 4 | the control-arm split does not move | **falsified** — it fell −0.0966, 1.3× its bar. See below, and read the limit with it |

**Prediction 3 deserves its explanation, because it was written as a trap and the
trap caught the wrong thing.** The pre-registration reasoned that an intervention
costing nothing is evidence it did nothing. P plainly did something — 3.3× its
bar — and cost nothing. The prediction conflated two quantities: removing
**label-level** dependence must cost performance, since baseline genuinely
predicts the label in a test set drawn from those labels; but removing only the
**amplification** — the part where the model used baseline *more than the labels
justify* — is free by construction, because over-use beyond the labels carries no
predictive power on a test set built from them. Costless was the correct
expectation for what P actually did.

#### 3.9c Prediction 4 — the split fell, and the construct behind it did not (2026-08-14)

The stage 7i harness on both `stage8-control` and `stage8-propensity`: **580
baseline-matched hosts x 3 periods = 1,740 rows per lane, 0 unscored**, the same
sizing as stage 7i. The draw is byte-identical to stage 7i's — both arms' fold
maps match `branches-20260808-rebaseline`'s exactly, so the seed-42 matcher
reproduces the same host set. Verified by `tic_id` checksum in pandas *before*
launching, rather than read out of the log afterwards.

| lane | F1 cut | pass | planet hosts | FP hosts | **split** |
|---|---:|---:|---:|---:|---:|
| stage 7i, `branches-20260808-rebaseline` | 0.4486 | 0.1943 | 0.2540 | 0.1345 | **+0.1195** |
| stage 8 control | 0.4047 | 0.2730 | 0.3575 | 0.1885 | **+0.1690** |
| stage 8 propensity | 0.3841 | 0.2201 | 0.2563 | 0.1839 | **+0.0724** |

**Prediction 4 is falsified.** Propensity against its own same-code control is
**−0.0966**: **1.3x** the 0.0720 split bar pre-registered in stage 7i, and
**1.6x** a 0.0592 paired bar computed on these hosts. A paired bootstrap over the
580 hosts gives 95% CI **[−0.157, −0.039]**, not crossing zero. Per the
pre-registered outcome table this reads as *"propensity weighting reduced
host-scoring as well as amplification. A second, independent win."*

**Running the control was load-bearing.** Stage 8's control splits at **+0.1690**,
not the +0.1195 stage 7i measured on the same hosts with the same code — a
**+0.0494** move from reseeding alone. Against the historical +0.1195 the
propensity arm's +0.0724 is −0.0471, *inside* the bar and readable as level. The
pre-registered comparison is against the same-code control, which is precisely
what the control arm exists for; had only the propensity arm been run, that drift
would have been credited to the intervention.

**The limit, recorded because omitting it would overstate the result.** The split
is a *thresholded* statistic and the arms do not sit at the same operating point.
Threshold-free, on the same hosts, the model's ability to tell a planet host from
an FP host on a transit-free light curve is **unchanged**:

| lane | host-AUC, transit-free | 95% CI |
|---|---:|---|
| stage 7i rebaseline | 0.5876 | 0.5429–0.6329 |
| stage 8 control | 0.6234 | 0.5812–0.6688 |
| stage 8 propensity | 0.6045 | 0.5566–0.6451 |

Paired over the 580 hosts, propensity minus control is **−0.0190, 95% CI
[−0.060, +0.019]**, crossing zero at p≈0.33 — while the split difference over the
*same* resamples does not. Propensity's scores on this population are shifted
down (median 0.187 against 0.248) and its F1 cut sits at the **78.0th** percentile
of them against the control's **72.7th**; a stricter operating point mechanically
shrinks a planet-minus-FP pass split.

**So the pre-registered statistic moved and the construct it stands for did
not.** This is post-hoc and does **not** overturn the pre-registered reading —
the same discipline stage 7i applied to its own common-cut diagnostic. What it
does is set the terms on which the win may be banked: **not until a
threshold-free measurement confirms it.** Recorded as a **qualified** second win,
and the qualification is not optional.

**The mechanism prediction 4 assumed does not hold either, and the reason is
worth keeping.** It reasoned that baseline dependence and host-scoring are
different defects because stage 7i showed the split survives baseline matching.
But matching removes the confound from the *host draw*; it does not remove the
*model's* learned reliance on baseline, and those are different operations. On
these matched hosts the control's score↔baseline correlation is **+0.0269** — the
matcher has already closed that channel (residual corr(baseline, label)
**+0.0452**), so there was nothing left there for the intervention to remove.
Propensity's is **−0.2528** on the same inputs while remaining **+0.3803** on the
evaluation slice: the arm's baseline response **changes sign between the two
populations**. Because the matcher's residual leaves planet hosts at a slightly
longer median baseline (1,259 d against 1,118 d), a negative dependence suppresses
planet hosts preferentially — consistent with the observed asymmetry, planet-host
pass **−0.1011** against FP-host pass **−0.0046**. These are transit-free
synthetic inputs and out of distribution, so the sign flip is not by itself a
defect; it is unexplained, and it is the most likely thing driving the split.

**One number here rests on a single draw.** The split's reseeding noise is
characterised by exactly one control retrain (+0.0494), and that is half the
effect being claimed — the same thinness already recorded against the three-draw
floors below.

Reproduction: `~/Downloads/.stage8-scratch/prediction4.py` rebuilds **the split
table and its bars** from the two `results/control_arm/stage8-*.parquet` files.
**It does not rebuild the threshold-free block** — the host-AUCs, either
bootstrap CI, the median/percentile diagnostics or the sign-flip correlations
have no recipe. *(Corrected 2026-08-15: this line previously claimed "every
figure in this subsection", which was false. The audit re-derived the
threshold-free block independently and it is correct — 0.5876 / 0.6234 / 0.6045,
paired −0.0190 with a CI crossing zero — but nothing in the repo re-derives it.)*

**Two defects in the run's own record, found while reading it.**

1. **`run_config` does not record the shard directory.** Arm N differs from the
   control only in which shard set it read, and the summary cannot say so — the
   two are distinguishable solely by `n_examples` (5,767 vs 5,426). A provenance
   gap: fix before any future arm selects its data by path.
2. **The measured floors came in roughly double stage 6's** — the control's
   pooled gate-recall floor is **0.0720** against stage 6's 0.0337. Three draws
   is a thin sd and it is being asked to carry decisions; worth widening before
   the next stage leans on it.

**Cost, for future sizing.** ~2 h per arm at `--n-models-per-fold 3` on this
machine, against the ~70 min the 2026-08-08 handover quoted — that figure is out
by ~1.7× and should not be used for planning. The prediction-4 harness came in
**at** its estimate for once: 48 min for the control arm and 49 for propensity,
1 h 37 for both, against the ~50 min per arm stage 7i recorded.

**The stage-3 incumbent summary needed no regeneration, and the reason it was
thought to is worth recording.** It was carried as outstanding on the grounds
that *"the label change invalidates it"*. Re-derived 2026-08-14: the summary
regenerates **byte-identical** to the committed
`models/cv/incumbent-rebaselined/cv_summary.json`, because the 2026-08-08 label
refresh flipped **zero** labels across the 5,703 `tic_id`s common to
`labels.previous.parquet` and `labels.parquet` — it added two rows and changed
nothing else, and neither added row is in the incumbent's scored set.

> **This no longer reproduces against the working tree, and the reason is not a
> defect in the claim.** Audited 2026-08-15: a catalogue refresh ran at 09:00 on
> 2026-08-15 and rotated `labels.parquet` into `labels.previous.parquet`, so the
> working tree now shows **5,705 common / 0 added**, which reads as a
> contradiction. Against the DVC pointer that was current when the claim was
> written the figures are **exactly** 5,703 common / 0 flipped / 2 added. The
> claim is correct; the artefact under it moved. See 5.4. **And had
labels moved, `summarise` could not have propagated it**: `load_predictions`
re-joins only `mission`, taking `y_true` from the predictions parquet, so the
label change would have needed `evaluate.py score`, not `summarise`. The
anticipated label change was group (a), which was never run. The deliverable
stands satisfied; the rationale attached to it did not survive checking.

**What stage 8 deliberately did not do — decided by Ollie, 2026-08-14.**

1. **External catalogue negatives (group a) were never started and will not be.**
   EB and brown-dwarf catalogues plus the ephemeris-match test, budgeted at up to
   8 h against an 8 h kill criterion. Arm P had already delivered the stage's
   reachable deliverable, so the marginal case was weak. **Recorded as
   deliberately not done, not as an oversight** — the negatives it would have
   produced remain available to any later stage that wants them.
2. **Arm S is not re-run.** Restoring the 680 dropped rows to the *test* split
   would make it comparable for ~2 h of compute, and the question it would answer
   has already been answered by P. The structural lesson stands in its place: **a
   resampling intervention has to keep the evaluation population whole even when
   it changes the training one.**

**Nothing promotes.** `models/registry.json` untouched; `ca906040` stays served.
