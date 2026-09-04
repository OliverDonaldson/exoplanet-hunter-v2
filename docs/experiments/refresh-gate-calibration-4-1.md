> Moved verbatim from `docs/roadmap.md` §4.1 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 4.1 Stage 10.5 — **CLOSED 2026-08-15**

Both halves are measured. Recall in **3.11c**, its floor corrected in **3.11d**,
the control-arm pass in **3.11e**. A3 was the last open prediction and is
falsified — on a premise nobody had checked rather than a prediction that failed.

Nothing promoted. The stage leaves two things for later items to answer:

1. **The served architecture is the more host-scoring one** (3.11e), which lands
   on W2 and on stage 11's serving decision, not on this stage.
2. **The control-arm split is not a trustworthy statistic** — it moved +0.15
   between two runs of one architecture that are threshold-free identical. Any
   later item planning to read it should read the host-AUC beside it, or instead.

### 4.1a Calibrate the refresh loop to its own noise — **CLOSED 2026-08-17**, all three defects

**New, 2026-08-15, from Ollie's question: does the weekly refresh apply the same
test each time?** The gate code is the same every run. The comparison is not.

**Three defects, all verified in code rather than inferred.**

1. **The evaluation population changes every refresh.** The candidate is scored
   on the new catalogue; the champion's stored summary was computed on the old
   one, so a ΔAUC blends the model effect with the population effect. The gate
   does detect this — `_population_mismatch` and `_gate_population_drift` raise —
   and 3.7 found it refusing to promote rather than compare mismatched rows.
2. **The folds differ every refresh.** Each run builds its own
   `StratifiedGroupKFold` over its own shard set, and `refresh_pipeline.py`
   passes no `fold_assignment`. `fold_sd` is non-zero in every summary, so part
   of any ΔAUC is only which rows landed where. **The capability to fix this
   already exists** — `9e0c0a5` built it for stage 10.5 — and is simply not
   wired in.
3. **The refresh has no measured floor at all.** `conf/train/default.yaml` sets
   `n_models_per_fold: 1` and the loop never overrides it, so a refresh candidate
   measures no reseeding spread. `decision_floor` returns `recall=None` and the
   gate falls back to `LEGACY_RECALL_TOLERANCE = 0.02` — the constant 3.7
   measured at **0.68σ**, under which a candidate whose true recall *equals* the
   champion's is rejected by noise about **31%** of the time, on the single
   criterion that has rejected every branch arm.

**What.** Raise `n_models_per_fold` on the refresh path so a candidate measures
its own floor; pin the fold assignment across refreshes so successive candidates
are comparable; keep a control lane so a promotion is attributable to the model
rather than the data.

**Why it is not cosmetic.** Until it lands, "promoted" and "rejected" from the
weekly loop are decisions taken against an uncontrolled comparison with a
tolerance already measured as too tight. Every deliberate result in this project
is read against a measured floor; the one automated decision is not.

**Deliverable.** A refresh run whose summary carries a real variance block, on
folds shared with its predecessor, and a gate decision quoting its own floor.

**Stops if.** The wider member count pushes a refresh past its window — in which
case pin the folds alone, which is free, and take the floor from a periodic
calibration run instead of every refresh.

#### Built 2026-08-16 — defects 2 and 3 closed, defect 1 still open

`da44ced`. The refresh now trains `N_MODELS_PER_FOLD` (default 3) so the gate
reads a floor **this run measured**, instead of falling back to the 0.68-sigma
constant. And the outer split is pinned across refreshes — but **rolling**, not
fixed, which is the part worth recording because both obvious options are wrong:

| approach | what breaks |
|---|---|
| rebuild the map each refresh | candidate and incumbent land on different partitions; part of every margin is only which rows fell where |
| pin one fixed map | uncovered groups are *dropped*, so every target the catalogue gains vanishes from training and the set shrinks weekly |
| **extend it** | neither — the shared population keeps its folds, new targets still enter |

`extend_fold_assignment` keeps every already-assigned group in the fold that has
always held it out, and places only new ones, into the fold currently holding
the fewest of their own class. **No group ever moves**, and the guard raises
rather than reconciling: a group that moved would be scored by a fold that had
trained on it, which is leakage wearing a split's clothes.

Verified on the real index — built over 5,426 groups at
`[1085, 1085, 1085, 1085, 1086]`; a simulated refresh adding 300 targets gives
**0 moved** and `[1146, 1145, 1145, 1145, 1145]`.

~~**Defect 1 is not fixed and is not wiring.**~~ **Closed 2026-08-17.** The
evaluation population itself grows every refresh, so a delta blended the model
effect with the data effect. Isolating it needed a **control lane** — the
champion re-scored on the new population — built at 4.1c, validated at 4.1d and
now **read by the weekly gate**: `refresh_pipeline` runs the lane after
`preprocess_and_shard` and passes its summary as `--champion-summary` on every
run. A weekly promotion is now attributable to the model.

Three things about the wiring worth having written down:

- **The reference follows the registry by construction, not by a branch.** The
  old `--champion-summary` had to be conditional — it named one model measured on
  one 2026-08-07 population, so passing it always would have frozen the
  comparison. The lane re-derives from `models/registry.json` every run, so
  passing it unconditionally is what keeps the reference moving.
- **There is no fallback, deliberately.** When the lane refuses, the flow does
  not gate at all: it reports UNRESOLVED and asks for a re-baseline. Substituting
  the stored summary for a control the lane declined to produce is exactly the
  "quietly deciding on the remainder" 4.1c rule 3 forbids. `incumbent-rebaselined`
  stays on disk because recorded commands name it, and nothing routes to it.
- **A refusal and a crash are different exits.** The lane exits `2` — UNRESOLVED's
  own code — when the shared slice is too thin, and non-zero-but-not-2 when it
  actually failed. This is the same conflation that made a crashed gate read as a
  quality rejection, fixed once here rather than twice later. Exit `0` with no
  summary written is also a failure: the file at that path would otherwise be
  last week's control, read as this week's.

#### Calibration run — 2026-08-16. It did not measure a floor; it found why there isn't one

`fc4f3515`, 5 h 28 for 5 folds x 3 members on the rolling map, trainer only so
nothing could reach the registry (`registry.json` verified byte-identical after,
`ca906040` still served).

**The run produced no recall floor, because the dual-view trainer does not write
one.** Raising `n_models_per_fold` was necessary and not sufficient:

| field | branch trainer | dual-view trainer |
|---|---|---|
| `seed_sd`, `fold_sd` | yes | yes |
| `recall_seed_sd`, `gate_recall_seed_sd` | yes | **no** |
| `pooled_gate_recall_seed_sd` | yes — 0.0310 on E-C | **absent** |
| `per_mission` block | yes | **absent** |

So `decision_floor(candidate).recall` is still `None` on the refresh path and the
gate still falls back to `LEGACY_RECALL_TOLERANCE = 0.02`. **Defect 1 of this
item is not fixed by the wiring.**

**And the gate cannot promote at all.** Run against this candidate, without
`--promote`:

> REJECT: gated on pooled CV means — a summary here predates the per_mission
> block; ROC-AUC 0.9633 vs incumbent 0.9581; Brier 0.0718 vs incumbent 0.0791;
> **populations differ: one summary carries no per_mission block, so the rows
> behind each mean are unknown; refusing to promote on a pooled comparison whose
> rows cannot be matched**

Exit code 1. The candidate is **better on every metric the gate could compare**:

| metric | candidate | incumbent | |
|---|---:|---:|---|
| CV ROC-AUC | **0.9633** | 0.9581 | better |
| Brier | **0.0718** | 0.0791 | better |
| ECE | **0.0240** | 0.0276 | better |

**This is 3.7's first defect, still live.** 3.7 found the gate could not promote
because the *incumbent's* stored summary has no `per_mission` block, and closed
it with `--incumbent-summary`. What was not noticed is that the **dual-view
trainer never writes one either**, so every refresh candidate has the same hole,
and `refresh_pipeline.py`'s `promotion_gate` task passes no such flag.

**Stated plainly: the weekly loop has never been able to promote, and still
cannot.** Its two outcomes are "rejected" and "rejected". A retrain that
genuinely beats the incumbent is refused on a provenance technicality, and the
refusal text is indistinguishable from a real quality rejection in the
notification. Every "promotion rejected" ping this project has sent may have
meant this instead.

**What closes it, and it is a build not a run.** The dual-view trainer needs the
branch trainer's summary schema: a `per_mission` block, and the recall variance
fields including `pooled_gate_recall_seed_sd`. Both already exist in
`train_branches.py::_aggregate_cv`; this is bringing one trainer up to the
other's contract, not inventing anything. Until then the calibration figure
below is all this run yields.

| measured, dual-view at n=3 on the rolling map | |
|---|---:|
| CV ROC-AUC | 0.9633 ± 0.0020 |
| AUC `seed_sd` (reseeding) | **0.0034** |
| AUC `fold_sd` (fold difficulty) | 0.0018 |
| AUC floor by stage 6's rule, `2 x sd / sqrt(3)` | **0.0040** |

The AUC floor is real and usable. The recall floor — the one the gate actually
reads, and the criterion that has rejected every arm — is still unmeasured.

#### The recall floor, measured — 2026-08-17. No retraining

**This closes 4.1a's calibration half.** `fc4f3515`'s member draws were on disk
the whole time: its `predictions.parquet` carries `member_score_0..2` over 5,375
rows. What was missing was a reader that turned them into the field the gate
reads — the trainer gained one in `adf4a71`, and `evaluate.py summarise`, the
tool for rebuilding a summary without retraining, still did not write a variance
block at all.

So `summarise_scored` now emits one, measured from the member columns already in
the prediction set. It is emitted whether or not there are draws to measure: a
block that appears only on success makes a missing key and a null read the same
to a person and differently to a program. With no member columns the draw count
is zero, which `decision_floor` reads exactly as it read a summary with no block
— so no existing decision moves, and re-summarising a single-model run such as
`incumbent-rebaselined` yields the same metrics it always did.

| measured on `fc4f3515`, pooled out-of-fold, n=3 | |
|---|---:|
| gate slice | TESS, 2,367 rows, 1,300 positive |
| TESS ROC-AUC | 0.9169 |
| TESS recall @1% FPR | 0.2569 |
| per-member pooled recall draws | 0.1885, 0.2623, 0.1600 |
| **`pooled_gate_recall_seed_sd`** | **0.05280** |
| recall floor, `2 x se(delta)` with the pooled prior | **0.0733** |

**The floor is 0.0733, and that is a large number.** It is 3.7x the
`LEGACY_RECALL_TOLERANCE = 0.02` the gate fell back to, and it means a shortlist
recall margin smaller than about 0.07 is not a difference this protocol can
resolve at three members. Read against it, the branch-arm rejections that were
decided on recall margins of a few hundredths were decided inside their own
noise — which is what 4.1a predicted would be found and is now measured rather
than asserted.

**Three draws remains a thin sd**, and its own sampling spread is roughly 40% of
its value. The floor is quoted with its `n` everywhere it appears, and a margin
comparable to it lands in UNRESOLVED rather than being read as either outcome.

The artefact is `models/cv/fc4f3515-resummarised/`, a sibling of the run rather
than a replacement for it — the same convention `incumbent-rebaselined` follows.
The original summary keeps its `folds` block and its AUC variance, neither of
which a pooled re-summary can reproduce.

### 4.1b Pre-registered — the gate's third verdict and what its floor is made of (pre-registered 2026-08-16; **implemented in `839ff8c`**)

Written before any of `adf4a71`'s follow-up is built, because two of the five
items change **what a verdict means** rather than adding a feature, and a
decision taken after seeing which runs flip is not a decision.

**Why this exists.** `adf4a71` let the gate read a dual-view candidate at all.
It did not make the gate safe to leave unattended: the verdict is a `bool`, the
tolerance scales with the candidate's own noise, and alarms are advisory in a
loop with nobody to advise.

#### 1. A third verdict — UNRESOLVED

Stage 6's **caveat 2** is binding and the gate cannot currently express it:

> *Three draws is a thin sd. The sampling spread of an sd from three draws is
> roughly 40% of its own value. If the result lands within a factor of ~1.5 of a
> threshold, the honest reading is **unresolved** — that is a stop-and-ask, not
> a re-specification.*

`PromotionDecision.promoted: bool` has two states for a rule with three. So a
margin at 0.8x the floor reads PROMOTE with the same confidence as one at 0.1x,
and a margin at 1.2x reads REJECT with the same confidence as one at 5x.

**Fixed now:** a criterion whose `|margin|` falls within **1.5x** of its floor —
i.e. `floor / 1.5 <= |margin| <= floor * 1.5` — is UNRESOLVED. Any criterion
UNRESOLVED makes the decision UNRESOLVED unless some other criterion already
REJECTs; **REJECT dominates UNRESOLVED dominates PROMOTE.** Unattended runs
treat UNRESOLVED as *do not promote, and say why it is not a rejection*.

*How it reads — fixed before re-gating anything.* `fc4f3515` currently reads
PROMOTE with recall margin **−0.0500** against floor **0.0610**, a ratio of
0.82. `0.0610 / 1.5 = 0.0407 <= 0.0500 <= 0.0915`, so it must become
**UNRESOLVED**. If it does not, this rule is not doing what caveat 2 says and the
implementation is wrong — not the caveat.

> **CORRECTION 2026-08-17 — the paragraph above describes a run state that has
> never existed on disk, and its falsification test is UNEXECUTED.**
>
> The pre-registration is left verbatim, as every pre-registration in this
> document is. What follows is the correction, not a re-specification.
>
> **What is actually true of `fc4f3515`.** Its summary carries no `per_mission`
> block, so `_gate_slice` returns None, `cand_recall` is None, and **the recall
> guard is skipped entirely — there is no margin and no floor to compare**. The
> gate REJECTs on population mismatch before reaching the criterion this
> paragraph is about. Re-gated read-only on 2026-08-17, the full verdict is:
>
> > REJECT: gated on pooled CV means — a summary here predates the per_mission
> > block; ROC-AUC 0.9633 vs incumbent 0.9558; Brier 0.0718 vs incumbent 0.0798;
> > populations differ: one summary carries no per_mission block, so the rows
> > behind each mean are unknown; refusing to promote on a pooled comparison
> > whose rows cannot be matched
>
> **Where the figures came from.** No summary on disk carries
> `pooled_gate_recall_seed_sd = 0.0528` or a TESS recall of 0.257; the 11 runs
> that measured the field span 0.0029–0.0632. The numbers are internally
> consistent with one specific artefact: a summary with
> `pooled_gate_recall_seed_sd = 0.0528` at `n = 3`, whose TESS recall of 0.2569
> sits 0.0500 below `incumbent-rebaselined`'s 0.3069. `2 x 0.0528 / sqrt(3)` is
> 0.0610 under the superseded rule and
> `2 x sqrt(0.0528^2/3 + 0.0353^2/3)` is 0.0733 under the adopted one, which is
> the pair of floors quoted here and in the handover. That artefact was
> `fc4f3515`'s summary **regenerated in memory** during the 2026-08-16 session
> and never written to disk — the handover records doing exactly that, and
> records that no dual-view summary on disk carries `per_mission`. So the
> figures are not fabricated; they were measured against a file that was not
> kept, which is why no reader can re-execute the test.
>
> **The defect, stated plainly.** A binding pre-registration was written in the
> present tense ("currently reads") about a state no one can reproduce. The
> three predictions under *What the floor is made of* below inherit this: they
> are read against the same absent artefact and are **equally unexecuted**. The
> handover's claim that "all three of 4.1b's predictions confirmed" is therefore
> not supported by anything on disk.
>
> **Not re-pointed at a different run.** Substituting a run that happens to be
> sliceable would be re-specifying a pre-registration to fit what is available.
> The test is executed once `fc4f3515` has a `per_mission` summary on disk —
> regenerated from its existing predictions, not retrained.
>
> **EXECUTED 2026-08-17.** `models/cv/fc4f3515-resummarised/` now holds that
> summary, rebuilt by `evaluate.py summarise` from the run's own
> `predictions.parquet`. Nothing was retrained. The reconstruction above is
> confirmed to the digit:
>
> | quantity | reconstructed | measured |
> |---|---:|---:|
> | `pooled_gate_recall_seed_sd` | 0.0528 | **0.052805** |
> | TESS `recall_at_1pct_fpr` | 0.2569 | **0.256923** |
> | margin vs `incumbent-rebaselined` (0.3069) | −0.0500 | **−0.0500** |
> | floor, superseded rule | 0.0610 | **0.0610** |
> | floor, adopted rule | 0.0733 | **0.0733** |
>
> Re-gated read-only against `incumbent-rebaselined`, the verdict is
> **UNRESOLVED** (exit 2), on the reason *"shortlist recall margin −0.0500 is
> within 1.5x of its 0.0733 floor — too close to call from three draws"*.
>
> **So the pre-registration's three predictions are confirmed, and its "How it
> reads" test passes — but only now, and against an artefact that had to be
> rebuilt to make it checkable.** What was falsified was never the arithmetic; it
> was the claim that the run *currently read* that way, when the file saying so
> had not been kept. The figures were right and unverifiable, which is the worse
> of the two failures because nothing looks wrong.

#### 2. What the floor is made of

**The defect.** `decision_floor` reads `pooled_gate_recall_seed_sd` from the
**candidate alone**. Across the 11 runs on disk that quantity spans
**0.0029 to 0.0632, a 22.1x range** (median 0.0353, mean 0.0381), so a noisier
candidate earns itself a wider band to clear. It also ignores the incumbent's
noise entirely, which stage 6's **caveat 1** already says is wrong: *"the
incumbent carries its own noise and a different architecture and protocol."*

**Three options were on the table. The choice is recorded before it is built.**

| option | why not |
|---|---|
| a floor pooled across runs | assumes the noise scale is a property of the statistic, not the architecture — which **caveat 4** flags as an assumption, not a finding. The 22x spread mixes architectures and interventions, so pooling would import branch-model noise into a dual-view decision |
| a fixed floor on a recalibration schedule | this is what the gate already had. `LEGACY_RECALL_TOLERANCE = 0.02` was fixed, went stale, and was measured at 0.68 sigma |
| **se of a difference** | **adopted** |

**Adopted: the tolerance is `2 x se(delta)`, where**

```
se(delta) = sqrt( sd_cand^2 / n_cand  +  sd_inc^2 / n_inc )
```

because the quantity under test **is a difference of two run means**, not one
run's mean. This is the only option of the three that is correct for what is
actually being compared, and it implements caveat 1 rather than restating it.

**The honest gap, recorded now rather than discovered later.** No incumbent
summary on disk carries a variance block — not `ca906040`'s, not
`incumbent-rebaselined`'s. So `sd_inc` is **unavailable today**. Until an
incumbent is re-baselined with one, the incumbent term uses the **pooled median
across the 11 measured runs, 0.0353**, and the decision text must say so in
words. A tolerance that silently substitutes a prior for a measurement is the
project's own recurring defect class.

**This does not remove the perverse incentive, and pretending otherwise would be
the error.** Under `se(delta)` a noisier candidate still earns a wider band —
correctly, because its mean is genuinely less well known. What stops that being
exploitable is **item 1**: a candidate noisy enough to make its margin
comparable to its own floor lands in UNRESOLVED, which does not promote. The two
items only work together.

*Predictions, recorded so they can be wrong.*

1. The recall tolerance on `fc4f3515` **widens** from 0.0610, because a second
   variance term is being added under a square root and nothing is removed.
2. It widens by less than `sqrt(2)`, since the pooled incumbent term (0.0353) is
   smaller than the candidate's own (0.0528).
3. The verdict is **UNRESOLVED either way** — a wider floor moves the ratio
   further below 1, deeper into the band, not out of it.

#### 3. The K2 alarm is permanent, and that is a decision not a bug

`incumbent-rebaselined` carries `['Kepler', 'TESS', 'all']` and **no K2**, so
"populations differ: only the candidate scored K2" fires on **every** future
candidate. An alarm that always fires is noise, and under item 3's strict mode it
would block every promotion forever.

**Recorded as permanently acknowledged, with the reason:** K2 entered training
after the incumbent was baselined, the gate decides on **TESS** alone, and K2's
absence from the incumbent cannot be fixed by anything the candidate does. The
alternative — re-baselining the incumbent on the K2-bearing population — is a
real option and is **not** taken here, because it changes the reference every
past result was read against. It is recorded as available if the pooled slice is
ever wanted for a decision.

**Nothing promotes on any of this.** `models/registry.json` is untouched and
`ca906040` stays served.

### 4.1c Pre-registered — the control lane, and what a weekly delta is allowed to mean (pre-registered 2026-08-17; **the lane is built** — its own check was FALSIFIED and is replaced by 4.1d)

Written before the lane exists, because it decides **what the one automated
number in this project refers to**, and a rule chosen after seeing which way the
first delta falls is not a rule.

**The defect it closes** is 4.1a's first, the only one still open. The candidate
is scored on the current population; the champion's stored summary was measured
on the population it was trained against. Every weekly ΔAUC and Δrecall is
therefore a model effect and a data effect added together, and the gate reports
the sum as if it were the first.

#### 1. What the lane does

Each refresh, before the gate runs, the **served champion is re-scored on the
current shard set** and summarised. The gate compares the candidate against that
fresh summary instead of a stored one. Nothing about the candidate changes.

#### 2. Which rows decide, and why not all of them

**The gating comparison is the shared out-of-fold population**: rows in both the
champion's training set and the current shard set, each scored by the fold that
held it out.

Measured today, before anything is built:

| population | rows | TESS | Kepler | K2 |
|---|---:|---:|---:|---:|
| champion `ca906040` trained on | 4,818 | | | |
| current shard set | 5,380 | | | |
| **shared — this is what gates** | **4,610** | **2,367** | 2,238 | 0 |
| added since the champion | 770 | 0 | 243 | 527 |
| dropped since the champion | 208 | | | |

**The 770 new rows are excluded from the gating comparison, and that is a
decision rather than an oversight.** The champion never trained on them, so
scoring them means averaging all five folds — an ensemble — while the candidate
scores each row with the single fold that held it out. That hands the champion a
five-model advantage on exactly the rows the refresh added. They are measured
and reported as a diagnostic slice, and they never gate.

**A consequence worth stating before it is observed:** the shared population is
fixed at the champion's training set and the current set grows, so the gating
subset ages. It does not shrink from below — it can only lose rows the catalogue
drops — but it covers a falling *fraction* of what the model serves. That is a
real cost of not re-baselining, and it is reported every run rather than
discovered later.

#### 3. The two deltas, and only one of them is the model

```
model effect  =  candidate(shared)  -  champion(shared)      same rows, two models
data effect   =  champion(shared)   -  champion(previous)    same model, two populations
```

The gate decides on the **model effect alone**. The data effect is reported
beside it and gates nothing — it is the quantity that has been silently inside
every weekly margin until now.

#### 4. How the result will be read — fixed before the lane runs

1. **The gate reads the model effect**, against the candidate's own measured
   floor by the rule already adopted in 4.1b. No new threshold is introduced
   here; this changes *what is compared*, not *how tightly*.
2. **A data effect larger than the model effect is reported in words** on every
   run, promoted or not. It does not block — a population that genuinely
   improved is not a fault — but a promotion taken while the data moved further
   than the model did is a promotion that has to say so.
3. **The lane refuses rather than narrows.** If the shared TESS slice falls below
   **1,000 rows**, the 1% FPR cut lands on fewer than ten negatives and the
   recall statistic is not worth reading. At that point the run reports
   UNRESOLVED and asks for a champion re-baseline. It does not quietly decide on
   the remainder.
4. **The champion's re-score must reproduce its own stored numbers on rows that
   have not changed.** If re-scoring the champion on the shared population
   disagrees with `incumbent-rebaselined` on the TESS slice by more than
   `1e-6`, the lane is measuring something other than what it claims and the
   result is void. This is the lane's own correctness check and it runs first.

*Predictions, recorded so they can be wrong.*

1. The champion's re-scored TESS slice reproduces `incumbent-rebaselined`'s
   exactly — same 2,367 rows, same fold assignment, same weights — because that
   artefact was produced this way. **If it does not, the lane is wrong, not the
   artefact.**
2. The model effect against `fc4f3515` is **smaller in magnitude** than the
   −0.0500 recall margin currently read against the stored summary, because that
   margin contains a data effect that this removes. Direction is not predicted.
3. The verdict on `fc4f3515` stays **UNRESOLVED**. Its margin is 0.7x its floor,
   and removing a component of the margin moves it further inside the band, not
   out of it.

**Stops if.** The re-score costs more than the refresh window allows. The
champion's five folds each score roughly 920 rows, so this is inference over
4,610 rows and not a retrain; if that estimate is wrong by an order of magnitude
the lane runs on a schedule of its own rather than every refresh, and the gate
falls back to the stored summary with the staleness reported.

**Nothing promotes on any of this.** `models/registry.json` is untouched.

#### Result — the lane is built, and its own correctness check is FALSIFIED (2026-08-17)

**Prediction 1 is falsified and rule 4 has fired, so by its own terms the
result is void. The criterion is not adjusted, and the lane is not wired into
the refresh flow. This is a stop-and-ask.**

**What was built.** `eval/control_lane.py` and `scripts/control_lane.py`. The
lane resolves the served run from the registry, re-scores it out-of-fold over
the current shard set, and summarises it. The shared population falls out of the
existing code rather than being computed separately: `score_run` under
`OUT_OF_FOLD` keeps only rows it has a held-out fold for, which is exactly the
incumbent's own training set intersected with today's shards.

**The compute estimate was wrong by three orders of magnitude.** 4.8 costed this
at 5–9 h. Measured: **10.6 seconds**, five folds over 4,610 rows, because it is
inference and not a retrain. The lane is cheap enough to run every refresh with
no scheduling argument at all — which is the one part of this that came out
better than pre-registered.

| measured, `ca906040` re-scored on the current shard set | |
|---|---:|
| shared population | 4,610 of 5,380 current rows (85.7%) |
| added since it trained | 770 — all K2 and Kepler, no TESS |
| dropped since | 208 |
| TESS gate slice | 2,367 rows, 1,300 positive |
| wall clock | 10.6 s |

**What falsified it.** Rule 4 required the re-score to reproduce
`incumbent-rebaselined`'s TESS slice to `1e-6`. It does not:

| metric | re-scored | stored | difference |
|---|---:|---:|---:|
| `roc_auc` | 0.910005 | 0.910004 | 7.2e-07 |
| `brier` | 0.121129 | 0.121128 | 7.6e-07 |
| `pr_auc` | 0.920337 | 0.920330 | **6.9e-06** |
| `ece` | 0.043828 | 0.043850 | **2.2e-05** |
| `recall_at_1pct_fpr` | 0.306923 | 0.306923 | 0 |

**The cause is identified and it is not floating point.** Two runs of the lane
are **bit-identical** to each other on every metric, so inference here is
deterministic. Against the stored artefact, **4,595 of 4,610 rows agree
exactly** and **15 differ**, one by 0.072. All 15 are TESS — 12 confirmed
planets and 3 false positives — and ~~their `period`, `duration` and `depth` come
from the label catalogue, which has been refreshed since the stored artefact was
produced on 2026-08-07. Changed parameters change the aux features, which change
the score.~~ **that mechanism is wrong and 4.1d disproves it.** `period`,
`duration` and `depth` are baked into the *frozen* shard index and cannot reach a
score at all; the only catalogue column `legacy_aux` reads live is `snr`, which
4.1d then confirmed by perturbation. The 15 rows and their dispositions are
correct as recorded — the account of *why* they moved was not.
`data/processed/tfrecords` itself is clean against its DVC pointer;
the three artefacts `dvc status` does report as drifted are the *viewset* ones,
already carried as Ollie's open decision in 4.8.

**So the check demanded something the lane exists to make false.** The lane
measures the incumbent on *today's* inputs; the stored artefact measured it on
7 August's. Requiring equality between them assumes the population never moves,
which is the assumption the control lane was built to stop making. The check is
mis-specified against its own subject.

**That is recorded, not repaired.** A pre-registered criterion is not adjusted
because executing it was inconvenient — that is the move this project's rules
exist to prevent, and the fact that the specification error is mine does not
create an exception. Predictions 2 and 3 are **not evaluated**: both are read
from a control summary that rule 4 declares void.

**What is Ollie's call.** The lane is committed and inert — nothing in
`refresh_pipeline.py` calls it, so Saturday's run is unaffected. Resolving it
means choosing what the correctness check should have been, and that is a
decision about what "the same measurement" means when the inputs are allowed to
move. It is not a decision to take by editing a tolerance until the run passes.

### 4.1d Pre-registered — the corrected check, which is two checks (pre-registered 2026-08-17; **run** — Check A passed bitwise)

Ollie's ruling on 4.1c. The original rule 4 **asked one question that was
actually two**, and mixing them is why it could not be answered. They are
separated here and written down before either is run.

#### Why this is a replacement and not a tolerance adjustment

Recorded explicitly, because a future reader looking for precedent to loosen a
criterion must not find one here.

**The tolerance does not move. It stays `1e-6`.** What changes is *what the
check ranges over*. The original demanded that a measurement taken on today's
inputs equal one taken on 7 August — which is true only if the population never
moves, and retiring that assumption is the entire reason the lane exists. The
rule this project runs on forbids **adjusting a criterion to fit a result**. It
does not forbid **replacing a criterion that measured the wrong thing**.

**The test of whether that distinction is honest: the corrected check can still
fail.** If the lane's calibrator, its aux pipeline, its fold assignment or its
metric definitions had diverged from the original path, then rows that did *not*
change would disagree too — on the same day, on identical inputs, with no
population drift available to blame. A repair that could not fail would be the
tolerance-shopping this is not.

**The restricted-population repair is available and is deliberately not taken.**
Keeping the original comparison and excluding the moved rows does work — the
evidence is already in hand at 4,595 of 4,610 — but it needs an exclusion
predicate defined **on the inputs** (rows whose catalogue fields were revised),
never on whether the output happened to match, or the test is defined by its own
result. Same-day equivalence needs no such care, so it is strictly the better
instrument. The restricted form is the fallback only if the original path cannot
be re-run, and Task 2B established that it can.

#### Check A — method equivalence. Time is removed from it

**The question.** Does the lane compute what the original path computes? That
has nothing to do with dates, so both sides are run **today**, against the same
frozen shard set, the same labels table and the same weights.

Left — the original path, the one that produced `incumbent-rebaselined`:

```
python pipeline/scripts/evaluate.py score \
    --run models/cv/ca906040cdb74ba6b07353a500244777 --protocol oof \
    --out results/champion_rebaselined_today.parquet

python pipeline/scripts/evaluate.py summarise \
    --predictions results/champion_rebaselined_today.parquet --protocol oof \
    --out models/cv/champion-rebaselined-today/cv_summary.json --exclude-unresolved
```

Right — the lane, checked against it:

```
python pipeline/scripts/control_lane.py \
    --out models/cv/control-lane/cv_summary.json \
    --reproduces models/cv/champion-rebaselined-today/cv_summary.json \
    --reproduces-predictions results/champion_rebaselined_today.parquet
```

**What must agree, at `1e-6`, with no exclusions and no coverage floor:**

1. **Every row.** Identical `tic_id` set, identical fold assignment per row, and
   `max |Δscore|` over all of them within tolerance. Not a sampled subset and not
   a slice — if the two paths score a row differently, that is the finding.
2. **Every slice.** Every metric of every `per_mission` block, not only the
   `TESS` one rule 4 looked at.

**One scope statement, made here so it is not a hidden carve-out later.** The
comparison is **out-of-fold only**, because out-of-fold is the only thing the
lane produces. The stored artefact bundles 4,610 out-of-fold rows with 770
zero-shot ones; the lane computes no zero-shot block at all, and `per_mission` is
built from held-out rows in both paths regardless. This excludes no row from a
comparison either side claims to make.

**If Check A fails, the lane is wrong and is not wired in.** That is a result,
not an obstacle, and the same rule applies to it as applied to rule 4.

#### Check B — the drift measurement. Not a gate, and not a failure

The 15 rows are the lane's **first real output**, not the wreckage of a failed
check. Fifteen rows moving in ten days — 0.33% of the gated population, 12
confirmed planets and 3 false positives, all TESS — is precisely the quantity
that justifies building the lane, and it is recorded as a measurement **with no
tolerance attached** and nothing riding on its size.

**One sub-measurement is owed, because 4.1c's stated mechanism is in doubt.**
That result attributed the drift to revised `period`, `duration` and `depth`
reaching the score through the label catalogue. Reading `legacy_aux` says that
cannot be the path: those fields are baked into the **frozen** shard index, and
the only catalogue column the scorer reads live is `snr`. Check B therefore
records *which input actually changed*, and 4.1c is corrected against whatever it
finds rather than left standing.

#### The staleness dependency — the one way this lane goes quietly wrong

`score_run` assembles each feature vector from **two sources with different
freshness**, and nothing anywhere checks that they correspond:

| what scoring reads | source | last moved |
|---|---|---|
| `global_view`, `local_view` | `data/processed/tfrecords/*.tfrecord` | **25 Jul** |
| `label` — the ground truth every metric is computed against | `tfrecords/index.parquet` | **25 Jul** |
| aux 0–6 and 8 — period, duration, depth and the rest | `tfrecords/index.parquet` | **25 Jul** |
| aux 7 — `snr` | `data/tables/labels/labels.parquet` | **15 Aug** |

Three consequences, all silent today:

1. **A revised disposition never reaches the lane.** Ground truth comes from the
   25 July index, so the lane scores against three-week-old labels while
   reporting itself as "the champion on the current population".
2. **A revised `snr` reaches it alone.** One feature moves while the eight beside
   it stay frozen — an internally inconsistent vector, and no guard notices.
3. **Rebuilding the view set moves everything at once**, including what a fold
   assignment refers to, and the lane would report that as a model measurement.

The lane's name claims more currency than its inputs have. **The missing guard
is a correspondence check between the shard set and the labels table**, and it is
recorded here as a known hole rather than closed in the same commit that
discovers it.

*Predictions, recorded so they can be wrong.*

1. **Check A passes** at `1e-6` on every row and every slice. The two paths call
   the same `score_run` and the same `summarise_scored`; what differs between
   them is a parquet round trip, a `protocol` column supplied by flag on one side
   and by assignment on the other, and the mission merge. If any of those is
   lossy, this is where it shows.
2. *(restated from 4.1c, still unevaluated)* The model effect against `fc4f3515`
   is **smaller in magnitude** than the −0.0500 recall margin read against the
   stored summary. Direction is not predicted.
3. *(restated from 4.1c, still unevaluated)* The verdict on `fc4f3515` stays
   **UNRESOLVED**.
4. **The drift is carried by `snr` alone**, and 4.1c's account of the mechanism
   is wrong.

**Stops if.** Check A fails: the lane does not get wired in, predictions 2–4 stay
unevaluated a second time, and the divergence is diagnosed before anything else
is built on the lane.

**Nothing promotes on any of this.** `models/registry.json` is untouched and
`ca906040` stays served.

#### Result — Check A passes bitwise, and the lane's first measurement is zero (2026-08-17)

**Check A: PASSED, and by more than the criterion asked for.** The two paths do
not agree to `1e-6` — they agree **exactly**, on every row and every metric.

| Check A, both paths run today on identical inputs | |
|---|---:|
| rows compared | 4,610, symmetric difference **0** |
| rows agreeing **bitwise** | **4,610 / 4,610** |
| max abs score difference | **0** (criterion: ≤ 1e-6) |
| fold assignments disagreeing | 0 |
| ground-truth labels disagreeing | 0 |
| metrics compared | 27, across `TESS`, `Kepler` and `all` |
| worst metric difference | **0** |

**What that does and does not establish.** It establishes that the lane's own
new code — resolving `fold_of` from the registry, the out-of-fold subsetting,
the mission merge, the summary assembly — introduces nothing. It **cannot**
detect a fault inside `score_run` or `summarise_scored`, because both paths call
them. That limit is a property of any equivalence check between two callers of
one primitive, and it is stated here rather than left for a reader to notice.

**Check B: the drift, measured.** 15 of 4,610 shared rows — **0.33%** — moved
beyond `1e-6` in the ten days since the stored artefact, worst **0.0718**. All
TESS, **12 confirmed planets and 3 false positives**. No row changed fold, and
**no row changed label**, which is itself a finding: ground truth comes from the
frozen index, so a disposition revised in the catalogue would not have shown up
here at all.

| data effect — same champion, two populations, TESS | lane today | stored 7 Aug | effect |
|---|---:|---:|---:|
| `roc_auc` | 0.910005 | 0.910004 | +7.2e-07 |
| `pr_auc` | 0.920337 | 0.920330 | +6.9e-06 |
| `brier` | 0.121129 | 0.121128 | +7.6e-07 |
| `ece` | 0.043828 | 0.043850 | −2.2e-05 |
| **`recall_at_1pct_fpr`** — the gating statistic | 0.306923 | 0.306923 | **exactly 0** |
| `recall_at_5pct_fpr`, `recall_at_10pct_fpr` | | | **exactly 0** |

**The mechanism, and 4.1c was wrong about it.** `legacy_aux` splices exactly one
live catalogue column — `snr` — into eight frozen ones; `period`, `duration` and
`depth` live in the 25 July index and cannot reach a score. Confirmed by
perturbation rather than argued: doubling `snr` on **exactly those 15 rows**
moved **exactly those 15 rows**, worst 0.0757 against the observed 0.0718, and
left the other **4,595 bitwise unchanged**. `snr` is the carrier, it is the only
carrier, and it has the right leverage.

**The 7 August values are gone and cannot be recovered.** `data/tables/labels.dvc`
has exactly one commit — 2026-08-15. Before that the labels table, **the only
live input the lane has**, was versioned by nothing. So the drift is measurable
and its direction is not: no one can reconstruct which `snr` values produced the
7 August artefact. This is a worse hole than the staleness dependency 4.1d
pre-registered, because it cannot be closed retroactively — only from here on.

*Predictions.*

| # | prediction | outcome |
|---|---|---|
| 1 | Check A passes at 1e-6 on every row and slice | **confirmed**, bitwise |
| 2 | model effect **smaller** than the −0.0500 stored margin | **FALSIFIED** — it is *equal*, at −0.050000 |
| 3 | verdict on `fc4f3515` stays UNRESOLVED | **confirmed** |
| 4 | the drift is carried by `snr` alone | **confirmed** by perturbation |

**Prediction 2 failed because its reasoning was wrong, and it should not have
been restated.** It assumed the −0.0500 margin contained a data effect the lane
would remove. The data effect on `recall_at_1pct_fpr` is **exactly zero** — the
15 moved rows do not cross the 1% FPR threshold — so the model effect is the
whole of it. Worse, 4.1c's own result table already recorded that recall
difference as `0`, so this was falsified by evidence in hand at the moment it was
carried forward into 4.1d unexamined. Restating a prediction is not the same as
re-deriving it, and this is what the difference costs.

**The honest headline: the lane's first measurement is that there was nothing to
correct.** It was built to strip a population effect out of the weekly margin;
on this candidate the population effect on the gating statistic measures zero, to
the last digit. Re-gated against the control summary, `fc4f3515` returns
**UNRESOLVED** with a **−0.0500** margin at **0.7×** the 0.0733 floor — the same
verdict, the same number, and the same reason as against the stored summary.

That is not an argument against the lane. Before this run, "the weekly margin is
contaminated by population drift" was an **assumption**, and 4.1c stated it as
the defect being closed. It is now a **measurement**, and on this occasion it
reads zero. The lane's value is that the next non-zero one will be visible
instead of silently inside the margin.

**Nothing promotes on any of this.** `models/registry.json` is untouched,
`ca906040` stays served, and every artefact written here is gitignored like every
other CV run.
