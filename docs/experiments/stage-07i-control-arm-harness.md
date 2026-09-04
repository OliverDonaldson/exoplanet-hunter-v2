> Moved verbatim from `docs/roadmap.md` §3.6 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.6 Stage 7i — the offline control-arm harness

#### 3.6a Pre-registered before the harness runs — recorded 2026-08-09, nothing built

The two limits already recorded under *Decided 2026-08-09* stand unchanged and
must survive into the result: **`detection`/`ghost` run masked** (`dv_usable`
False — no DV report exists at a synthetic ephemeris), and **this does not
restore comparability with 26.4%**, so the dual-view incumbent is re-measured
through the same harness and the comparison is made within protocol.

Reading the code surfaced three further choices the sweep never had to make.
Fixed here, before the harness exists.

**1. Fold routing is out-of-fold, and a host that cannot be routed is dropped.**
Control-arm hosts are real labelled TESS targets that were in training, so the
only honest protocol is the fold that held each one out — read from the run's own
`predictions.parquet` `fold` column, which carries it. A host absent from that
frame is **dropped, not zero-shot scored**: silently mixing the two protocols
across one population is the exact defect stage 3 closed, and `eval/scoring.py`
already refuses to guess between them.

**2. The pass threshold is the run's own recall @1% FPR operating point, and
both operating points are reported.** A branch run directory stores **no**
threshold — `bundle["threshold"]` is the legacy *serving* bundle's field, and the
branch bundle carries `calibrator`, `platt_a`, `platt_b`, `scalar_constants` and
nothing else. So one must be chosen rather than read:

| candidate | verdict |
|---|---|
| **1% FPR on the run's own out-of-fold TESS rows** | **primary.** It is the threshold every gate decision in this project is made at, and "passes threshold" then means "would reach the shortlist" |
| mean of the folds' F1-optimal thresholds | **reported alongside.** It is what the live path used for the original 26.4%, so omitting it would discard the only continuity available |

Both are applied to the same scores, so reporting the pair costs nothing and
stops the choice of operating point from being an unexamined degree of freedom.
The same rule is applied to the incumbent, computed from its own re-baselined
predictions.

**3. Baseline matching is on `baseline_days`, by quantile stratum, drawn equally
per label.** The statistic of interest is not 26.4% but the **46.7% / 12.3**
split between planet hosts and false-positive hosts — and observation baseline is
the confound that split is most exposed to, at +0.387 against the label on TESS.
So hosts are stratified into `baseline_days` quantile bins and drawn equally from
each label within each bin; strata that cannot supply both labels are dropped and
**the number dropped is reported**, because a matcher that quietly discards its
hard strata reports a clean number about an easier population.

`labels.parquet` has **no `expected_transit_count`** (18 columns, confirmed), so
`baseline_days` — `(expected_transit_count − 1) × period` — is unavailable there
and comes from a join against the viewset scalars.

**The comparison population, measured 2026-08-10 before the run.** Hosts routable
out-of-fold in **both** run directories — an unrestricted pair would compare two
different populations again, which is the defect this project keeps hitting.

| | hosts |
|---|---:|
| cached TESS, depth-filtered | 1,058 |
| routable in `branches-20260808-rebaseline` | 1,057 |
| routable in `ca906040` | 1,052 |
| **routable in both — the comparison population** | **1,051** *(552 FP / 499 planet)* |
| **after baseline matching, 4 strata** | **580** *(290 per label, 0 strata dropped)* |

The per-stratum counts are the +0.387 correlation made concrete, and they are why
matching caps the draw at 580:

| baseline stratum | FP hosts | planet hosts | matched pairs |
|---|---:|---:|---:|
| 0 *(shortest)* | 230 | 33 | 33 |
| 1 | 164 | 99 | 99 |
| 2 | 81 | 181 | 81 |
| 3 *(longest)* | 77 | 186 | 77 |

**580 hosts x 3 periods = 1,740 rows per lane** is therefore the full measurement,
and it is the pool's maximum rather than a chosen sample size.

**Recorded as unmeasured, not assumed away.** Matching costs sample size, and the
cached-FITS host pool is what it is. If a matched draw cannot fill its strata the
result is reported at the size it reaches, with the unmatched rate beside it —
not topped up from unmatched hosts.

#### 3.6b Pre-registered before the measurement — recorded 2026-08-10, nothing run

Both lanes exist and are smoke-tested; **no control-arm number has been produced
for either model.** Written down first because nobody is watching an autonomous
session read its own result.

**What is run.** Two invocations, identical hosts: `--per-stratum 200`,
`--seed 42`, three periods (3, 7, 12 d), each restricted with
`--also-routable-in` to the other run. That yields the same 580-host matched
draw for both — 290 per label, 4 strata, none dropped — so the comparison is
paired on the host.

**The criterion, re-specified as unmeasurable-as-written on 2026-08-09 and
settled here.** "26.4% must fall" cannot be read against the live figure. What
is tested instead:

> **Does the branch model score the *star* less than the dual-view incumbent, on
> one common offline protocol?**

**The bar, computed before the numbers exist.** A pass rate is a proportion over
hosts, and the three periods of one host are correlated, so the effective n is
the **host** count and not the 1,740 rows. At n=580 a single rate carries
`2σ ≈ 0.04`; at n=290 per label, `2σ ≈ 0.06`; the planet-minus-FP split combines
in quadrature to `2σ ≈ 0.07`. The comparison between models is **paired on the
same hosts**, which makes an unpaired bar conservative — it is used anyway, and
the pairing is noted rather than banked.

**How each outcome reads — fixed now.**

| outcome | reading |
|---|---|
| branch rate **below** incumbent by more than 0.04 | the branch architecture scores the star less, **on this protocol only**. It cannot support a serving claim, and it is not a promotion argument — the gate is AUC and shortlist recall |
| within **±0.04** | **level.** The branch architecture does not reduce host-scoring. Given stage 4 rejected every arm on recall, this is the outcome that closes the branch line's remaining case |
| branch rate **above** incumbent by more than 0.04 | the branch architecture scores the star *more*. Report it; do not re-specify |

**The split is the sharper statistic and is read alongside.** The headline 26.4%
conflates two populations; the 46.7 / 12.3 gap is what "the model scores the
star" actually predicts. A model that vets the transit should show a **smaller
planet-minus-FP split**. Because hosts are baseline-matched, a residual split
can no longer be explained by observation baseline — which is the whole reason
the matcher exists.

**Predictions, recorded so they can be wrong.**

1. Both models pass **well under 26.4%** at the shortlist cut, because that cut
   is far stricter (1% FPR, ~0.96–0.97 calibrated) than the F1-optimal cut the
   26.4% was measured at. The F1 cut is the one to compare against 26.4% loosely.
2. Both show a **positive** planet-minus-FP split at the F1 cut — the pathology
   is real and matching removes the baseline confound, not the effect.
3. The two models land **within 0.04 of each other** on the overall rate. Stage
   4 found the branch line level on TESS AUC and worse on recall; there is no
   measured reason to expect a large control-arm separation.

**Nothing promotes.** `models/registry.json` untouched, `ca906040` stays served.
A favourable number here does not reopen stage 4.

#### 3.6c Result — the branch architecture does not score the star less (2026-08-12)

`results/control_arm/`. **580 baseline-matched hosts x 3 periods = 1,740 rows per
lane, 0 unscored**, exactly the sizing pre-registered before the run. Both limits
recorded on 2026-08-09 survive into the result and are written into the output
JSON: `detection`/`ghost` ran masked, and these numbers cannot support a claim
about *serving*.

**Pass rates, with the host as the inferential unit.** The rates below are the
driver's row-level output over 1,740 rows; the bars beneath them use the **host**
count of 580, because a host's three periods are correlated and the row count
would overstate the precision. *(Corrected 2026-08-12: an earlier version of this
line said a host's three periods are averaged before the population mean. They
are not — no averaging happens anywhere in the driver. Row-level and host-level
agree to every digit here only because every host contributed exactly three
scored rows and none were dropped; they diverge as soon as
`n_unscored_dropped` is non-zero, which the driver permits. Recorded rather than
left to hold by luck.)*

| | cut | pass | planet hosts | FP hosts | **split** |
|---|---:|---:|---:|---:|---:|
| incumbent, shortlist | 0.9623 | **0.0000** | 0.0000 | 0.0000 | +0.0000 |
| branch, shortlist | 0.9731 | **0.0006** | 0.0000 | 0.0011 | −0.0011 |
| incumbent, F1 | 0.5390 | 0.1230 | 0.1839 | 0.0621 | **+0.1218** |
| branch, F1 | 0.4486 | **0.1943** | 0.2540 | 0.1345 | **+0.1195** |

2σ bars: **0.036** on a single rate (n=580), **0.051** per label (n=290),
**0.072** on the split. Paired on the host, `2×se = 0.0344`.

**The pre-registered PRIMARY operating point returned a floor, and that is a
finding about the protocol rather than a result.** At the 1% FPR cut *both*
models pass essentially nothing — 0.0000 and 0.0006. There is no room for a rate
to fall, so the comparison that was nominated as primary **cannot carry the
stage's criterion**. It is reported as level because it is, and because the
alternative — quietly promoting the secondary point to primary after seeing the
numbers — is the thing pre-registration exists to prevent. The F1 point was
pre-registered as *reported alongside*, so it is available without
re-specification.

**At each model's own operating point the branch model scores the star MORE.**
Paired over the same 580 hosts, `+0.0713` against a paired bar of `0.0344` —
**2.1x the bar**. Per the pre-registered outcome table this reads as *"the branch
architecture scores the star more. Report it; do not re-specify."*

**The split — the sharper statistic — is level.** +0.1195 against +0.1218, a
difference of **−0.0023 against a 0.072 bar**, which is 3% of it. The 46.7 / 12.3
pathology is reproduced in both models at essentially identical magnitude, on
hosts where observation baseline has been matched away. Whatever drives
host-scoring, **eleven diagnostic branches do not reduce it.**

**Three predictions, two right and one falsified.**

| # | prediction | outcome |
|---|---|---|
| 1 | both pass well under 26.4% at the shortlist cut | **correct**, and by more than expected — both ≈0 |
| 2 | both show a positive planet-minus-FP split at F1 | **correct** — +0.1218 and +0.1195 |
| 3 | the two models land within 0.04 on the overall rate | **FALSIFIED at the F1 point** (+0.0713). Correct at the shortlist point, where the floor makes it trivial |

**A diagnostic, explicitly NOT pre-registered and not part of the reading.**
Scored at the *incumbent's* cut rather than its own, the branch model passes
0.1075 against 0.1230 (paired −0.0155, inside the bar) with a split of +0.0747.
So much of the +0.0713 is where each model's F1 optimum sits — the branch
model's is lower (0.4486 vs 0.5390) — rather than host-scoring alone. This is
post-hoc, it does **not** overturn the pre-registered reading, and it is recorded
because omitting it would overstate the result.

**Verdict: stage 7's criterion is NOT met.** "The 26.4% control-arm host-pass
rate must fall", re-specified on 2026-08-09 as a within-protocol comparison, is
answered **no**: the split is level and the own-operating-point rate is higher.
Combined with five arms rejected on shortlist recall, **the branch line now has
no measured advantage on either of its two criteria.** That is the outcome the
pre-registration named as closing its remaining case.

**Two defects were found and fixed while running this, and the numbers above are
post-fix.** Both are the house failure mode — a plausible number from a broken
computation:

1. **Score collapse across periods.** `score_through_run` assigned by `tic_id`
   match, but `write_viewset_shards` permutes rows, so positional alignment was
   unavailable and every row of a host matched. All three of a host's periods
   were overwritten with whichever was processed last — two thirds of the
   measurement silently discarded, with a wholly plausible pass rate surviving.
   Fixed by aligning against the index the writer actually produced, with a
   raise if the stream and index disagree. The incumbent lane never had this bug,
   which is how it surfaced: its per-host scores differed and the branch lane's
   did not.
2. **The stream was iterated four times per fold** — once per member plus once
   to read identities back. Folded into the members' own passes. Scoring went
   from **~10 minutes per fold to ~14 seconds**.

**Cost, for future sizing.** Build ~50 min for 1,740 rows (dominated by two BLS
periodograms per row; the multi-sector tail is much slower than the median),
scoring ~1 min. The incumbent lane is ~10 min end to end. `--shard-dir` keeps the
built shard set on disk instead of a temp dir, so a failed scoring pass leaves
something to inspect and re-score by hand. *(Corrected 2026-08-12: it does **not**
skip the build on a re-run — an earlier version of this line said it did. The
driver rebuilds unconditionally, on purpose: a directory left by a different host
draw, seed or period list would otherwise be scored as though it were this one,
which is a wrong measurement wearing a plausible pass rate. Thirty-five minutes
is the cheaper side of that trade.)*

**Nothing promotes.** `models/registry.json` untouched; `ca906040` stays served.
