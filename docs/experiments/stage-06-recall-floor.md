> Moved verbatim from `docs/roadmap.md` §3.4 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.4 Stage 6 — the recall noise floor

#### 3.4a Pre-registered before the run — recorded 2026-08-08, run not yet launched

Nobody is watching an autonomous session read its own result, so every number
below is written down first.

**The decision rule, taken from the AUC precedent rather than invented here.**
The adopted AUC threshold is `2 x seed_sd / sqrt(n_models_per_fold)`:
`2 x 0.0081 / sqrt(3) = 0.0094`, which is the recorded "a margin under ~0.009 is
not a decision". Applied unchanged to recall, a margin `M` **is not a decision**
when

```
per-member seed_sd  >=  M x sqrt(3) / 2
```

**The two numbers, computed before the run exists.**

| question | margin | the rejection/effect survives only if per-member `seed_sd` is |
|---|---:|---:|
| Is run 3's rejection sound? *(TESS R@1%FPR 0.145 vs incumbent 0.307)* | 0.162 | **below 0.1403** |
| Was the capacity arm's 0.145 -> 0.236 noise? | 0.091 | at or **above 0.0788** for it to be noise |

`pooled_gate_recall_seed_sd` is the estimator these are read against.
`gate_recall_seed_sd` is reported beside it and is an upper bound, so a
conclusion that survives *it* is safe a fortiori.

**How each outcome reads — fixed now.**

| `pooled_gate_recall_seed_sd` | reading |
|---|---|
| **< 0.0788** | run 3's rejection is sound, and the capacity arm's recall jump is **outside** reseeding noise. It stays **unactionable** regardless — the architecture it was measured on does not exist on HEAD — so it is recorded as a lead for stage 7, not built on |
| **0.0788 – 0.1403** | run 3's rejection stands; the capacity jump is **inside** noise and is retired as an observation. This fully discharges the queued capacity repeat |
| **>= 0.1403** | **run 3's margin does not support its own rejection.** Report as falsified, stop, and surface it. Do not re-specify the criterion. This does not un-reject run 3 — AUC, Brier and the gate's other guards are untouched — but it would mean the criterion this project has leaned on cannot carry the weight put on it, which is a finding |

**Prediction, recorded so it can be wrong.** `pooled_gate_recall_seed_sd` lands
in **0.02–0.08**, and `gate_recall_seed_sd` lands above it. The re-baseline's own
`per_mission.TESS` AUC lands **within ±0.009 of run 3's 0.9119**, and its TESS
recall @1% FPR lands in **0.10–0.30** — a deliberately wide band, because a
narrow one on the statistic whose spread has never been measured would be false
precision.

**Four caveats, recorded now rather than discovered later.**

1. **Reseeding noise is necessary, not sufficient.** 0.145 vs 0.307 compares two
   *different models*, not two seeds of one; the incumbent carries its own noise
   and a different architecture and protocol. Clearing the threshold does not by
   itself make the rejection sound. Failing it does make it unsound.
2. **Three draws is a thin sd.** The sampling spread of an sd from three draws is
   roughly 40% of its own value. If the result lands within a factor of ~1.5 of a
   threshold, the honest reading is **unresolved** — that is a stop-and-ask, not
   a re-specification.
3. **Two TESS populations, as always.** The pooled draws are over the run's own
   `per_mission.TESS`, **n=2,399**. The 0.145 and 0.307 in the table above are
   the **n=2,367** shared-join gate slice. Both are correct; neither is being
   "fixed" to match the other.
4. **The floor is measured on HEAD, applied to run 3's margin.** That assumes the
   noise scale is a property of the statistic and the population rather than of
   the specific architecture. Both are eleven-branch models on the same shards,
   so it is reasonable — and it is an assumption, not a measurement.

**What will be run, named before it exists.** One CV run to
`models/cv/branches-20260808-rebaseline`, `--n-models-per-fold 3`, 5 folds, over
`data/processed/viewset_tfrecords` unchanged. Then, as a **reported diagnostic
and not a decision**, `evaluate_promotion` against
`models/cv/incumbent-rebaselined/cv_summary.json` — the same protocol run 3's
table used, so the two are comparable. `promotion_gate.py` itself is not used:
it reads the registry, the live incumbent has no `per_mission` block and it has
no `--incumbent` flag.

**The tree condition, stated precisely rather than as "clean".** No *tracked*
file will differ from the recorded `git_sha` — `git status --porcelain
--untracked-files=no` is empty at launch. `run_config.git_dirty` will
nevertheless be **True**, because `--porcelain` counts untracked files and
`docs/demo-script-2026-08-08.md` is one: written by another session at 21:10,
read by nothing, and not this session's to commit. Recorded here so the flag on
the control run is explained by the artefact rather than guessed at later. Run 3
recorded no provenance at all and the capacity arm recorded `dirty: True` with
no explanation; this is the first branch run whose flag means something specific.

*(Outcome: the run recorded `git_dirty=False`, not True. The stated reason for
predicting True stopped holding between writing this and launching — a broad
`git add docs/` in `6005506` swept that untracked file into the commit, so the
tree really was clean. The provenance claim is stronger than pre-registered, but
it is stronger by accident, and the prediction was wrong for a reason worth
recording rather than quietly benefiting from.)*

**Nothing promotes.** The re-baseline is a control. `models/registry.json` is
untouched, `ca906040` stays served, and a favourable number here does not reopen
stage 4 — only stage 7's leave-one-out runs, read against this re-baseline, can
move anything.

#### 3.4b First launch failed on fold 0; the pre-registration is unchanged — 2026-08-09

The run died ~15 minutes in, in `fit_platt`, on the convergence guard added
2026-08-08. The guard was right: `nll` clipped `p` while the analytic gradient
did not, so BFGS's line search failed on an inconsistency between `f` and `jac`.
Fixed by the stable `softplus(z) - y·z` form; the guard is untouched. Full
detail in the audit.

**It implied nothing for run 3, and that was checked rather than assumed.** It
was recorded here first that runs predating the guard had "calibrated on a
stalled fit" — an inference from the guard's absence, not a measurement.
Regenerating run 3's calibration from its own checkpoints reproduces its stored
scores at **max |delta| = 0.0**, and refitting with the converged optimiser
returns the same parameters: **run 3's Brier and ECE stand exactly as recorded**
(TESS 0.1150 / 0.0171). The defect needs validation rows saturating past `_EPS`,
and run 3 had **2 in 4,344** against a score of exactly 1.0 in the fold that
stalled. See the audit for the per-fold table and for what remains unmeasured.

**None of the pre-registered numbers move, and this is why.** Every quantity the
thresholds above are read against is **rank-based**, and Platt is monotone:
`pooled_gate_recall` is computed from the *uncalibrated* `member_score_*`
columns and never touches a calibrator at all, and per-mission ROC-AUC and
recall @1% FPR are invariant to any monotone rescaling. The calibration fix
moves **Brier and ECE only**. The 0.1403 and 0.0788 thresholds, the three-way
outcome table and the predictions stand exactly as written before the first
launch — they are not being re-specified after a failure, and nothing about the
failure touched a result.

If the re-baseline's TESS AUC lands outside ±0.009 of 0.9119 in either direction,
that is a **falsified prediction** and is reported as one. It is not re-read as
an improvement or a regression: one run of a control cannot carry that, and stage
7 re-baselines again anyway.

#### 3.4c Result — the recall noise floor, measured (2026-08-09)

`models/cv/branches-20260808-rebaseline`, HEAD `6005506`, `git_dirty=False`,
5,426 rows, `--n-models-per-fold 3`. **Every pre-registered prediction landed
within its band**, and the outcome is **branch 1** of the three fixed before the
first launch.

**The deliverable.** Recall @1% FPR now has the error bar AUC has had since
2026-08-08:

```
pooled gate draws (n=2,399 TESS rows):  0.1857  0.1842  0.2355
pooled_gate_recall_seed_sd = 0.0292          <- the primary estimate
```

| floor | seed_sd | fold_sd |
|---|---:|---:|
| AUC *(existing; run 3 had 0.0081 / 0.0094)* | 0.0060 | 0.0101 |
| recall, all missions | 0.0485 | 0.0547 |
| recall, gate slice, **per fold** | 0.0689 | 0.0888 |
| recall, gate slice, **pooled** | **0.0292** | — |

The fold-level estimate is **2.4× the pooled one**, exactly as pre-registered —
a fold's TESS slice carries ~215 negatives and a 1% FPR cut of two rows, so it
was only ever an upper bound. Using it would have overstated the noise by more
than a factor of two.

**The threshold this yields**, by the same rule AUC's came from
(`2 × seed_sd / √n_models`):

> **A recall @1% FPR margin under ~0.034 is not a decision.**

**The two pre-registered questions, answered.**

| question | margin | vs 0.0337 | verdict |
|---|---:|---|---|
| Is run 3's 0.145-vs-0.307 rejection sound? | 0.162 | **4.8×** the floor | **sound** |
| Was the capacity arm's 0.145 → 0.236 noise? | 0.091 | **2.7×** the floor | **not noise** |

The capacity arm's recall jump was a **real effect**, not the artefact this
project feared. Per the pre-registration it stays **unactionable regardless** —
`init_filters=22` was measured on an architecture that is not on HEAD — so it is
recorded as a lead for stage 7 and nothing is built on it.

**The re-baseline itself** *(its own `per_mission` block, n=2,399 on TESS — not
the n=2,367 gate slice)*:

| slice | n | AUC | Brier | ECE | R@1% FPR |
|---|---:|---:|---:|---:|---:|
| K2 | 527 | 0.9351 | 0.1000 | 0.1141 | 0.1683 |
| Kepler | 2,500 | 0.9547 | 0.0839 | 0.0275 | 0.5328 |
| **TESS** *(gates)* | 2,399 | **0.9202** | 0.1108 | **0.0159** | **0.2196** |
| all | 5,426 | 0.9365 | 0.0973 | 0.0108 | 0.2855 |

Against the two runs on the identical shard set:

| | TESS AUC | TESS R@1% FPR |
|---|---:|---:|
| run 3 | 0.9119 | 0.1434 |
| capacity arm | 0.9076 | 0.2332 |
| **re-baseline** | **0.9202** | **0.2196** |

**Shortlist recall rose 0.1434 → 0.2196 against run 3 — a margin of 0.0762, or
2.3× the floor measured in the same run.** Read with pre-registered caveat 1:
reseeding noise is *necessary, not sufficient* for a cross-run margin, and this
one spans three training-path changes at once (unfolded rebuild, stratified inner
split, augmentation masking), so it attributes to none of them. It is also still
well below the incumbent's 0.307.

**Gate: REJECT**, run as a reported diagnostic and not a decision, exactly as
pre-registered. TESS AUC 0.9202 vs 0.9100 and ECE 0.0159 vs 0.0438 both favour
the candidate; **shortlist recall 0.220 vs 0.307 rejects it**, which is the same
criterion that closed stage 4 and it is now the first time that rejection is
backed by a measured floor. `ca906040` stays served; the registry is untouched.

**The Kepler alarm, explained as the pre-commitment requires.** Kepler fell
−0.0367 against the incumbent, past the 0.02 alarm. It is not a regression: at
0.9547 this is the **best Kepler any branch model has reached** (run 3 0.9464,
capacity 0.9449), so the gap narrowed rather than widened. The branch line has
lost on Kepler since run 1 and the cause is unresolved — the narrow-span,
high-count cell that three architectures have not moved.

**Two readings that are not decisions, recorded because they will be tempting.**
The +0.0083 TESS AUC over run 3 sits at 92% of its own pre-registered band and
just under the AUC floor's 2σ (~0.007 at this run's `seed_sd 0.0060`) — **level
is the honest reading**, not an improvement. And observation baseline sensitivity
rose to **+0.3025** (run 3: +0.2901), now *above* the +0.278 label correlation
itself; transit sensitivity moved further from zero at **−0.1467** (run 3:
−0.1249). Both are reported diagnostics, both point at stage 8, and neither gates
anything.
