> Moved verbatim from `docs/roadmap.md` §3.11 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.11 Stage 10.5 — the ensemble arm

#### 3.11a Pre-registered — recorded 2026-08-12, nothing run

**Why this exists.** Five arms have been rejected, every one on shortlist recall,
and every one asked the same question: *does this replace the incumbent?* Nobody
ever asked whether it **complements** it. Measured on the 2,367 shared TESS
gating rows, at each model's own 1% FPR cut:

| caught | n |
|---|---:|
| both | 117 |
| incumbent only | 282 |
| branch only | 172 |
| neither | 729 |

Spearman agreement between the two scores is **0.654**. They are not a better
and a worse model; they are right about different targets. Combining them:

| combiner | TESS AUC | recall @1% FPR |
|---|---:|---:|
| branch alone | 0.9215 | 0.2223 |
| incumbent alone | 0.9100 | 0.3069 |
| mean of probabilities | 0.9498 | 0.4292 |
| **mean of logits** | **0.9537** | **0.4746** |
| rank-average | 0.9535 | 0.4538 |

**Every combiner beats both models.** Mean of logits is +0.168 recall over the
incumbent — **5.7× the 0.0337 floor** — and needs no population statistics, so it
is deployable as written.

**This is exploratory and is NOT a result.** One draw, no ensemble variance
estimate; the two runs used different CV splits, so while no row's own label
leaked (both scores are out-of-fold in their own run), it is not a clean joint-CV
measurement. **A pre-registered confirmation run is required before any of it is
read as a finding.**

**Sequenced after stage 8, on the same argument that put stage 8 ahead of stage
9** — stage 8 changes the labels both models learn from, so measuring this first
means measuring it twice. Ollie's decision, 2026-08-12.

**Pre-registered now, before the run exists.**

*What is run.* Both models on a **common fold assignment**, the ensemble scored
out-of-fold, `--n-models-per-fold 3` so the ensemble carries its own variance
estimate rather than borrowing a single model's.

*What is measured.* TESS AUC and recall @1% FPR on the gate slice; the
control-arm split through the stage 7i harness; and Spearman(score,
`baseline_days`), because an ensemble that inherits the *worse* of its two
members' baseline dependence is not an improvement for the deployment use.

*The bar.* Recall @1% FPR against the incumbent's 0.307, read against the
ensemble's own measured floor by the stage 6 rule. Nothing under **1×** the floor
is a decision.

| outcome | reading |
|---|---|
| ensemble recall **above** incumbent beyond its floor | the branch line's value is as a **complement**, not a replacement. This reopens nothing about stage 4 — those rejections were about replacement and remain correct |
| **within** its floor | the disjointness is real but does not convert into shortlist recall. Record it and close the branch line as the plan already anticipates |
| ensemble recall **below** the incumbent | the exploratory reading was an artefact of the mismatched splits. Report it as falsified; do not re-specify |

*Predictions, recorded so they can be wrong.*

1. The confirmation run lands **below** the exploratory 0.4746, because the
   mismatched-split version gave each model a slightly different training set
   and that flatters an ensemble.
2. It still clears the incumbent's 0.307 by more than its floor.
3. Baseline sensitivity of the ensemble sits **between** its two members'
   (+0.5155 branch, +0.3812 incumbent) rather than below both — averaging does
   not remove a confound both models share.

*Nothing promotes on this either.* A favourable ensemble number is an argument
for a serving change, which is stage 11 work and Ollie's call.

#### 3.11b Amendment — recorded 2026-08-14, before anything was built or run

**The pre-registration above stands verbatim.** This amendment settles four
things it could not have anticipated, and it is recorded *before* the first line
of the build so that none of it can be chosen after seeing a number.

**1. Two ensemble arms, not one. Ollie's decision, 2026-08-14.** Stage 10.5 was
sequenced after stage 8 on the argument that *"stage 8 changes the labels both
models learn from"*. **It did not** — group (a) was skipped and the labels never
moved (3.9b). What stage 8 changed is the training *weighting*, and arm P is the
only thing the stage delivered. Which branch model enters the ensemble is
therefore an open question the pre-registration does not answer, and picking one
silently would confound the answer. Both run, against **one shared dual-view
member on the same folds**:

| arm | branch member | what it is for |
|---|---|---|
| **E-C** | the plain control, un-weighted | the like-for-like confirmation of the exploratory 0.4746 |
| **E-P** | the propensity-weighted arm | the one carrying stage 8's deliverable |

*How the pair reads — fixed now.*

| outcome | reading |
|---|---|
| **both** arms clear their bar | the complement finding is robust to the weighting. Carry **E-P** forward, since it also carries stage 8's amplification fix |
| **E-C** clears, **E-P** does not | propensity weighting costs the ensemble what it gained the single model. A real trade-off, reported as one — it does **not** retract stage 8, whose result is on a different statistic |
| **E-P** clears, **E-C** does not | the exploratory reading was specific to the un-weighted branch model. Report it, and say plainly that the 0.4746 was not the thing confirmed |
| **neither** clears | the disjointness is real but does not convert into shortlist recall. Close the branch line as 4.8 anticipates |

**2. The bar is the ensemble's own dual-view member, not the incumbent's 0.307.
Ollie's decision, 2026-08-14.** The pre-registration says *"recall @1% FPR
against the incumbent's 0.307"*. That figure is `ca906040` on **its own folds and
its own rows**. The common-fold dual-view is a different model trained on a
restricted population, so measuring the ensemble against 0.307 would blend the
ensemble effect with the refit effect — the confound stage 8's control arm exists
to prevent, arriving in a new place. Stage 8 has just demonstrated what that
costs: its control moved +0.0494 on a statistic through reseeding alone.

**So: ensemble recall @1% FPR against the common-fold dual-view member's own
recall, on the same folds, read against the ensemble's own measured floor by the
stage 6 rule. The incumbent's 0.307 is reported beside it as the historical
figure and does not gate.** This changes what the number is measured *against*,
not how it reads; both outcome tables stand as written.

**3. The evaluation population is the 5,375 tics both shard sets carry.**
`data/processed/tfrecords` holds 5,380 examples and `data/processed/viewset_tfrecords`
5,426; the intersection is **5,375**. Rows outside it are dropped from CV in both
models, so each trains on a slightly smaller set than its solo run. That is the
price of a joint measurement, recorded now rather than discovered in the reading.

**4. The threshold-free host-AUC is reported beside the control-arm split.**
Prediction 4 established that the split moves with operating-point placement
independently of the construct it stands for (3.9c). The split remains the
pre-registered statistic; the host-AUC over the same hosts, with a paired
bootstrap, is reported alongside. An addition to the reporting, **not** a
re-specification of the bar.

**What has to be built before any of this can run, found 2026-08-14.**

1. **Neither trainer accepts an external fold assignment.** `training/train.py`
   and `training/train_branches.py` each construct their own
   `StratifiedGroupKFold` over their own shard set, and the sets differ, so no
   seed makes them agree. A shared fold artefact plus injection into both is the
   blocking build. It is **reusable** — stages 9 and 7ii face the same cross-run
   comparability problem.
2. **The dual-view trainer has no `n_models_per_fold`.** The pre-registration
   requires `--n-models-per-fold 3` *"so the ensemble carries its own variance
   estimate rather than borrowing a single model's"*, and only
   `train_branches.py` supports it. Either it is built on the dual-view side, or
   the ensemble ships without the variance estimate its own bar depends on —
   which would make the bar unreadable, so it is built.

*Predictions for the second arm, recorded so they can be wrong.*

1. **E-C and E-P land within each other's floor** on recall @1% FPR. Propensity
   weighting moved amplification without moving AUC or recall on the single
   model, and there is no measured reason for an ensemble to behave differently.
2. **E-P's baseline sensitivity sits below E-C's**, because one of its two
   members has had its amplification removed — but **both sit above the dual-view
   member alone**, since averaging cannot remove a confound both members share.
   This is prediction 3 of the original pre-registration, applied per arm.
3. **Neither arm moves the control-arm host-AUC** off the ~0.60 that stage 8 left
   it at. Host-scoring has now survived every intervention aimed at it.


#### 3.11c Result — the ensemble confirms, on both arms (2026-08-15)

Three CV runs on the common fold assignment, `--n-models-per-fold 3`: one
dual-view member shared by both arms, and one branch member per arm. **The joint
measurement is joint** — all three agree on which fold holds each of the 5,375
tics, 0 mismatches, checked against the pinned map from three independent code
paths before any ensemble number was formed.

| model | TESS AUC | recall @1% FPR |
|---|---:|---:|
| dual-view, common folds — **the bar** | 0.9187 | **0.3046** |
| branch, E-C *(un-weighted)* | 0.9250 | 0.2831 |
| branch, E-P *(propensity)* | 0.9165 | 0.2000 |

| arm, mean of logits | TESS AUC | recall @1% FPR | margin vs its dual-view member | floor | |
|---|---:|---:|---:|---:|---|
| **E-C** | 0.9549 | **0.4362** | **+0.1315** | 0.0340 | **3.9x** |
| **E-P** | 0.9527 | **0.4223** | **+0.1177** | 0.0285 | **4.1x** |

Every combiner beat the dual-view member on both arms; mean of probabilities and
rank-average land below mean of logits, as the exploratory reading found.

**Both arms clear their bar, which per 3.11b reads as: the complement finding is
robust to the weighting, and E-P is the one to carry forward because it also
carries stage 8's amplification fix.** This is the first positive result the
branch line has produced. **It reopens nothing about stage 4** — those five
rejections were about *replacement*, they were correct, and this is a claim about
*complement*.

**The floors are the ensembles' own**, formed draw by draw — ensemble draw `i` is
dual-view member `i` combined with branch member `i` — not either member's floor
borrowed. That is what `n_models_per_fold` on the dual-view trainer was built
for; without it this table would have no bar to be read against.

> **The `3.9x` and `4.1x` in the table above are falsified in their stated
> form.** The member pairing they rest on was never pre-registered, and it is
> the pairing that minimises the floor. Audited 2026-08-15; see **3.11d**, which
> supersedes those two multipliers and the `3.8x` in prediction 2 below. **The
> recall numbers, the margins, and the finding itself are unaffected.**

**Three of five predictions confirmed, one falsified, one confirmed in part.**

| # | prediction | outcome |
|---|---|---|
| 1 | the confirmation lands **below** the exploratory 0.4746 | **confirmed** — 0.4362. The mismatched-split version did flatter it |
| 2 | it still clears the incumbent's 0.307 by more than its floor | **confirmed** — +0.1293, 3.8x |
| 3 | ensemble baseline sensitivity sits **between** its two members | **confirmed for E-C, FALSIFIED for E-P** — see below |
| A1 | E-C and E-P land within each other's floor on recall | **confirmed** — 0.0139 apart against floors of ~0.03 |
| A2 | E-P's sensitivity below E-C's, both above the dual-view member | **confirmed** |

**Prediction 3 fails in the direction nobody proposed.** For E-C the ensemble sits
between its members (dual-view **+0.3880**, ensemble **+0.4756**, branch
**+0.4938**). For E-P the members are **+0.3880** and **+0.3956** — and the
ensemble is **+0.4240, above both of them.** Two models each less
baseline-dependent than E-C's branch, combined, produced something *more*
baseline-dependent than either. The prediction's stated form — *between* — is
falsified. Its reasoning, that averaging cannot remove a confound both models
share, is vindicated harder than it was written: averaging did not merely fail to
remove the confound, **it manufactured more of it.** Unexplained, and it is the
first mechanism in this project seen to *create* baseline dependence rather than
inherit or amplify it.

**A diagnostic, not pre-registered.** E-P's branch member alone scores 0.2000
recall against E-C's 0.2831, which invites the reading that propensity weighting
costs shortlist recall — unlike stage 8, where it cost nothing. The drop is
**0.77x E-P's own gate-recall floor**, so it is **level**, not a demonstrated
cost. Recorded because the raw gap is the more quotable number and it is the
wrong one.

**Nothing promotes.** A favourable ensemble number is an argument for a serving
change, which is stage 11 work and Ollie's call. `models/registry.json` untouched;
`ca906040` stays served.

**Cost, measured.** Dual-view **4 h 44** for 5 folds x 3 members; each branch arm
**~2 h 02**. Nine hours for the three, against the 8 h estimated and the 12-14 h
this session briefly projected off the dual-view's pace alone. **The two
architectures differ by more than 2x per run and should be sized separately.**

**The floors remain thin, and this stage leans on them.** E-C's
`gate_recall_seed_sd` is **0.0677** and E-P's **0.0935**, against stage 6's
0.0337 — the same doubling recorded in 3.9b, now carrying a headline result.
Three draws is a thin sd. The margins here are 3.9x and 4.1x, so the conclusion
survives a considerably wider floor, but the next stage to lean on this quantity
should widen it first.

#### 3.11d The floor's pairing was never pre-registered — the multipliers are falsified (2026-08-15)

**Found by audit**, not by the session that produced 3.11c.

**The defect.** 3.11c's floor is formed from three ensemble draws, where draw `i`
pairs dual-view member `i` with branch member `i`. **That pairing appears nowhere
in 3.11a or 3.11b.** It exists only in the docstring of an untracked scratch
script. Both trainers seed members `seed * 1000 + i`, but the same integer on two
different architectures over two different shard sets produces statistically
independent draws — so member `i` on one side has no correspondence to member `i`
on the other, and the pairing is arbitrary. With three members there are `3! = 6`
equally defensible pairings, each giving a different floor.

**Per the standing rule — a result outside its pre-registration is reported as
falsified, never re-specified — the `3.9x` and `4.1x` are falsified in their
stated form.** So is the `3.8x` in prediction 2, which divides by the same floor.
What is *not* falsified: the recall figures, the margins, the AUCs, the baseline
sensitivities, and the finding that both arms clear their bar. Those are
independent of the pairing and reproduce exactly.

**Full disclosure of what was already known when the rule below was fixed.** This
is a re-analysis, not a fresh experiment, and pretending otherwise would be the
same error one level up. At the moment the rule was written the audit had already
computed, for both arms, the **minimum and maximum** floor over all six pairings,
and had established that the margin clears `1x` under **every** one of them. The
only quantity still unknown was the mean over the six. The rule is therefore
fixed against an outcome that is already largely visible, and it is recorded that
way rather than dressed as a blind pre-registration.

*The replacement rule, fixed 2026-08-15 before the mean was computed.*

1. The pairing is arbitrary, so the floor **marginalises over it**: the reported
   floor is the **mean of the six per-pairing floors**, each computed by stage
   6's rule, `2 x sd(draws) / sqrt(3)`.
2. The **maximum**-pairing floor is reported beside it as the conservative bound.
   **The finding is banked only if the margin clears `1x` under the maximum**,
   not merely under the mean.
3. The minimum-pairing floor is reported too, and is explicitly **not** the
   headline, because it is the one 3.11c happened to use.
4. This changes what the margin is divided by. It does **not** re-open the
   outcome table in 3.11b, which is keyed on clearing the floor, not on the size
   of the multiplier.

*Predictions, recorded so they can be wrong.*

1. Both arms still clear `1x` under the **maximum**-pairing floor, so the
   complement finding is banked unchanged. *(Already known to be true when
   written — recorded for completeness, not as evidence.)*
2. The mean-pairing floor sits **nearer the midpoint of the six than either
   extreme** for both arms, i.e. the identity pairing is an outlier rather than
   typical. This one is genuinely open.
3. **E-P's spread across pairings is wider than E-C's**, because its branch
   member's own `gate_recall_seed_sd` is the larger of the two (0.0935 against
   0.0677), so which member it pairs with matters more.

**Nothing promotes on this either**, and it changes no serving decision.

##### Result — the finding is banked, on a floor that no longer depends on a choice

| arm | margin | min | **mean** | max | x (mean) | x (max, the bar) | x (min, *not* the headline) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E-C** | +0.1315 | 0.0340 | **0.0407** | 0.0523 | **3.2x** | **2.5x** | 3.9x |
| **E-P** | +0.1177 | 0.0285 | **0.0469** | 0.0691 | **2.5x** | **1.7x** | 4.1x |

**Both arms clear `1x` under the maximum-pairing floor, so per rule 2 the
complement finding is banked.** The headline multipliers are **3.2x and 2.5x**,
against the **3.9x and 4.1x** 3.11c reported. Every recall figure, margin, AUC and
baseline sensitivity in 3.11c is unchanged — only the divisor moved.

**The size of the error is worth stating plainly.** The pairing 3.11c used was the
floor-minimising one on both arms, and on E-P it understated the floor by
**1.6x** against the mean and **2.4x** against the max. On a margin this large it
changes nothing. On a stage that landed at 1.5x it would have been the whole
result.

**All three predictions confirmed.**

| # | prediction | outcome |
|---|---|---|
| 1 | both arms clear `1x` under the max-pairing floor | **confirmed** — 2.5x and 1.7x. *Known when written; not evidence* |
| 2 | the mean sits nearer the midpoint of the six than either extreme | **confirmed** for both — the identity pairing is an outlier, not typical |
| 3 | E-P's spread across pairings is wider than E-C's | **confirmed** — 0.0406 against 0.0184, tracking its larger `gate_recall_seed_sd` |

**Prediction 2 of 3.11a re-reads as still confirmed**: E-C clears the incumbent's
0.3069 by +0.1293, **3.2x** the mean floor and **2.5x** the max, against the
`3.8x` 3.11c recorded.

**What this says about the three-draw floors, now recorded a fourth time.** With
three members the pairing choice moves E-P's floor by **2.4x** end to end. That
sensitivity is a direct consequence of estimating an sd from three draws, and it
is a second, independent reason to widen them before another stage leans here.

Reproduction: `~/Downloads/.stage8-scratch/floor_marginalised.py`, which writes
`floor_marginalised.json` beside itself.

#### 3.11e Result — the control-arm pass, and the architecture nobody had measured (2026-08-15)

The measurement 3.11a pre-registered and 4.1 carried: the control-arm split
through the stage 7i harness, plus the threshold-free host-AUC 3.11b added
beside it. Three lanes on the **identical** 580-host draw — 290 planet / 290 FP,
1,051 routable in both runs, `tic_id` checksum matching stage 7i's — confirmed in
pandas before any compute was spent, as 4.1 required.

The ensemble's score is `sigmoid(mean of logits)` of its members' calibrated
scores: the combiner 3.11c reported, put back on the probability scale the
harness thresholds on. Monotonic, so it changes no ranking statistic. **Its
operating points are derived from the ensemble's own out-of-fold predictions,
not borrowed from a member** — the same argument 3.11b made for the floor.

| lane | F1 cut | pass | planet | FP | **split** | **host-AUC** |
|---|---:|---:|---:|---:|---:|---:|
| dual-view alone | 0.4009 | 0.5448 | 0.6805 | 0.4092 | **+0.2713** | **0.7102** |
| branch E-C alone | 0.4016 | 0.2655 | 0.3379 | 0.1931 | +0.1448 | 0.6184 |
| branch E-P alone | 0.5001 | 0.1276 | 0.1506 | 0.1046 | +0.0460 | 0.5626 |
| **ENSEMBLE E-C** | 0.5737 | 0.1649 | 0.2448 | 0.0851 | **+0.1598** | **0.7338** |
| **ENSEMBLE E-P** | 0.5804 | 0.1339 | 0.2103 | 0.0575 | **+0.1529** | **0.6928** |

**A3 is falsified, and it is the premise that was wrong rather than the
prediction.** A3 said *neither arm moves the control-arm host-AUC off the ~0.60
stage 8 left it at*. Nothing moved it — both ensembles sit inside their dual-view
member's interval (E-C +0.0236, 95% CI [−0.008, +0.055]; E-P −0.0174, 95% CI
[−0.052, +0.016], both crossing zero). But the level is **not ~0.60**. It is
~0.71, because the dual-view member puts it there.

**The ~0.60 was only ever the branch architecture's number.** Every control-arm
measurement this project has taken — stage 7i, stage 8's arms, both branch lanes
here — was a *branch* model, at 0.56 to 0.62. The dual-view architecture, **the
one being served**, had never been read against them until now:

| architecture | host-AUC, transit-free | paired d vs dual-view | 95% CI |
|---|---:|---:|---|
| **dual-view, stage 10.5 common folds** | **0.7102** | — | — |
| **incumbent `ca906040`, dual-view** | **0.7123** | −0.0020 | [−0.020, +0.018] *crosses* |
| branch E-C | 0.6184 | **+0.0919** | **[+0.029, +0.153]** *excludes* |
| branch E-P | 0.5626 | **+0.1477** | **[+0.086, +0.215]** *excludes* |

**Two independently trained dual-view runs, on different folds and a different
population, land 0.0020 apart.** That is not one run's fluke. The gap to either
branch model excludes zero. **The served architecture is materially the more
host-scoring one, and this project has spent its control-arm budget measuring the
other one.**

**This reframes the branch line a second time, and W2 with it.** Stage 4 rejected
the branch architecture on shortlist recall, and those rejections stand. But on
the defect W2 names — *the model scores the star, not the transit* — the branch
models are **better than the incumbent**, by a margin excluding zero on 580
matched hosts. 3.11c found the branch line's value as a complement on recall;
this is a second, independent axis, and on it the branch line is not
complementary but superior.

**The split's unreliability is now demonstrated rather than argued.** The
incumbent splits at **+0.1218** and the stage 10.5 dual-view at **+0.2713** — a
gap of +0.1495 — while their host-AUCs differ by 0.0020. Same architecture, same
hosts, indistinguishable on the threshold-free construct, and the pre-registered
statistic disagrees by more than the entire effect stage 8 reported. 3.9c warned
that the split moves with operating-point placement independently of the
construct it stands for. This is that warning, measured.

**So stage 8's qualified second win should not be banked.** 3.9c set the terms:
not until a threshold-free measurement confirms it. The threshold-free
measurement it already had crossed zero, and the split is now shown to move by
+0.15 between two runs of one architecture that are threshold-free identical.
**The qualification is not optional, and the case for banking has weakened rather
than strengthened.**

**What this does not say.** These are transit-free synthetic inputs and out of
distribution. Host-AUC here is a *defect* measure, not a performance one — higher
is worse. Nothing here bears on which model ranks real candidates better, and
nothing here promotes. A serving change is stage 11 work and Ollie's call.

**Two build defects, both found by running rather than reading.** Both were
hard-coded assumptions that broke the moment the dual-view trainer numbered its
checkpoints, and both blocked the measurement rather than corrupting one.

1. `run_kind` matched the exact filename `cnn_dualview.keras`, so a multi-member
   dual-view run could not be scored **at all**. The branch lane has globbed
   since stage 4 and was never affected, which is how the asymmetry survived a
   whole stage. Fixed in `4e8c90a`.
2. `build_host_dualview` hard-coded the incumbent's 9-dim aux width against a run
   trained on 13. It raised inside sklearn **after 12 min 35 s of view
   building**. The width is now read from the run's own calibration bundle before
   the build starts. Fixed in `f87ef83`.

`47c4e61` predicted this shape exactly — *"one bug of that shape was found;
assume it was not the only one"* — and there were two more.

**Cost, measured.** Branch lanes 49 m 30 s and 49 m 25 s; the dual-view lane
**12 m 21 s**, far cheaper because it needs no shard round-trip. 1 h 51 for all
three, against ~2.5 h estimated.

Reproduction: `~/Downloads/.stage8-scratch/analyse_41.py`.
