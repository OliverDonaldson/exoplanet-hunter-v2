> Moved verbatim from `docs/roadmap.md` §3.5 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.5 Stage 7 — the attribution sweep, and the criterion problem

#### 3.5a Stage 7's criterion is blocked on stage 11 — found 2026-08-09

**The control-arm host-pass rate cannot be measured for any branch model today,
and nothing recorded this.** `roadmap.md` gates stage 7 on "26.4% must fall",
measured on real hosts with no injection. That number comes from
`injection_recovery.py`, whose control arm is `snr_target == 0, depth 0` pushed
through the **live scoring path**. That path cannot carry a branch model:

- `ScoringEnsemble.from_registry` loads `cnn_dualview.keras` per fold and
  predicts from `global_view`, `local_view`, `aux_features`.
- `TargetScorer` builds those with `preprocess.views.build_views` — the
  dual-view builder, not `preprocess.viewset`'s eleven.
- The runner constructs `TargetScorer(models_dir=Path("models"))` and reports
  `scorer.ensemble.run_id`: it scores **whatever the registry serves**, with no
  flag for a CV run directory. Pointing it at an ablation by editing
  `models/registry.json` is a hard non-negotiable.

Adding a run-directory flag is necessary and nowhere near sufficient — the
serving path would still build the wrong views. **Making a branch model scoreable
from a light curve is stage 11's "`TargetScorer` computes every branch live"**,
budgeted 10–15 h and scheduled four stages later. Stage 7 as written depends on
it.

Separately, `select_hosts` filters on mission, label, depth and cache only:
**the baseline-matched harness owed since old 2(b) does not exist**, so even with
serving parity the hosts would not be matched on observation baseline.

Three ways out were put to Ollie: pull stage 11's serving parity forward; build
an **offline** injection/control harness; or re-scope and defer.

#### 3.5b Decided 2026-08-09: the offline harness, with the incumbent re-measured on it

**Why not pull serving parity forward.** Stage 8 already invalidates stage 7's
attribution numbers — that is why stage 8 was moved ahead of stage 9 — so
spending 10–15 h on serving parity to obtain a stage-7 number stage 8 obsoletes
is paying for it twice, the same argument in the same direction. And serving
parity earns its keep in stage 11 by *shipping*; pulled forward it does the same
work under another stage's pressure and ships nothing. What attribution actually
needs is cross-ablation comparability on **one fixed protocol**, which the
offline harness gives directly.

**It is cheaper than it first looked, because the chain is existing parts.**
`build_view_set(lc, *, period, t0, duration, trend_lc=None, raw_lc=None)` builds
all eleven views from a light curve and an ephemeris with **no DV products**. So:
`clean_lightcurve` / `flatten_lightcurve` (already composed in `host_baseline`) →
`inject_box_transit` (model-independent, already tested) → `build_view_set` →
`write_viewset_shards` → `make_viewset_dataset` → the run directory's fold
members and calibrator, loaded exactly as `train_branches.py` does. New code is a
baseline matcher, a driver and the pass-rate report: one script, one test file.
Compute is forward passes over cached FITS — trivial beside a 2 h CV run.

Writing a shard and reading it back through `make_viewset_dataset` is the
construction rather than a shortcut: it puts the control arm through the same
parse and scalar-normalisation path training used.

**Two things pre-registered now, before the harness exists.**

1. **The offline control arm measures the model minus its DV path.** It injects
   at a synthetic ephemeris with depth 0, so no DV report exists for that
   ephemeris and the `detection` / `ghost` branches run masked
   (`dv_usable=False`). The model handles that by design — it is what the gating
   is for — but it is a real difference from how 56% of training rows were built,
   and it is a property of the measurement, not a defect to discover later.

2. **This does NOT restore comparability with 26.4%, and "must fall" is
   re-specified.** The 26.4% came through the live **dual-view** path: different
   model, different view builder, different preprocessing. An offline
   branch-model control arm is a **new baseline, not a continuation**. For "must
   fall" to be a test rather than a slogan, the **dual-view model is re-measured
   through the same offline harness** and the comparison is made within protocol.
   That is cheap — its checkpoints are on disk.

   This is a criterion being re-specified, and the reason is attached
   deliberately: it is re-specified as **unmeasurable as written**, *before any
   result exists*, rather than adjusted after seeing one. Those are different
   acts and only the second is the thing pre-registration exists to prevent.

**The honest limit, recorded so the number is not over-read later.** This can
never support "the serving defect is fixed" — only "the branch architecture has a
lower control-arm rate than the incumbent **on a common offline protocol**". If
the offline and live protocols disagree, that gap is itself a finding, and it is
only obtainable at stage 11.

**Also settled: the sweep's control is valid.** The re-baseline predates the
branch-drop refactor, so its checkpoints could in principle have been built by
different code. Checked rather than assumed — `branches-20260808-rebaseline`'s
`fold_0/model_0_cnn_branches.keras` against HEAD's no-drop build:

```
params 169,361   layers 134   inputs 13
ordered (layer name, layer class) sequence: IDENTICAL
```

Identical creation order means an identical initialiser RNG draw order, so the
six hours of sweep are not carrying an uncontrolled architecture difference.

*(Stated as the ordered sequence rather than as a digest of it. A digest is only
a fact if the function that produced it is recorded too — two correct checks of
this ran and returned different hashes because they hashed differently, which is
exactly the trap. The comparison above is reproducible from this repo.)*

**Stage 7 *(old D)* — branch attribution.** Which of the twelve branches earn
their place, now that the unfolded one can actually see a transit. A branch-drop
mechanism driven by config rather than by editing `build_cnn_branches` — leaving
a branch out must be a declared experiment, not a code edit — plus the harness
owed since old 2(b) was specified: **injection-recovery on matched hosts with
observation baseline held constant**, the only causal measure available here and
immune to the label-selection confound that moved to stage 8. **Twelve branches
over eleven views group into eight families** — the "eleven branches / ~7
families" this file carried until 2026-08-09 counted views, not branches, and
`BRANCH_FAMILIES` is now derived so the two cannot drift again. One leave-one-out
CV run each at `--n-models-per-fold 3`.
**Its criterion is already specified and is not AUC**: the control-arm host-pass
rate, 26.4%, must fall — **and is currently unmeasurable; see above.**

#### 3.5c Pre-registered before the sweep — recorded 2026-08-09, nothing launched

**Scope: three families, not eight, prioritised by uncertainty rather than
completeness.** A full sweep spends hours confirming that `flux` and `global`
matter, which is not in doubt.

| family | why it is uncertain | params |
|---|---|---:|
| `unfolded` | rebuilt 2026-08-08 (`65d1d98`) and never attributed; the branch stage 7 exists for | −8.5% |
| `periodogram` | built and never measured; largest non-flux capacity delta, and the most plausible redundancy with flux | −14.4% |
| `scalar_only` | tests whether the scalar path earns the `dv_usable` masking complexity | −9.8% |

**This sweep is EXPLORATORY, and that is fixed now rather than after the
numbers.** It reads AUC and recall @1% FPR deltas — which is what stage 4 did
four times without persuading anyone — because the criterion that would settle
the stage is blocked. **No branch is added or removed on the strength of it.** A
family that passes earns a confirmatory repeat, not a decision.

**The threshold is a two-run difference, not a margin against a fixed number.**
Both the control and each ablation are 3-member ensembles, and their errors are
independent, so the spread of the *difference* combines in quadrature:

```
sd(delta) = sqrt( (sd_control/√3)² + (sd_ablation/√3)² )
```

Using the re-baseline's measured floors, and assuming the ablation's own floor is
comparable, the 2σ bars are **recall ≈ 0.048** and **AUC ≈ 0.0098** — *not* the
0.034 and 0.007 that apply to a margin against a fixed reference. Each run
reports its own `pooled_gate_recall_seed_sd` and `seed_sd`, so the bar is
recomputed per comparison rather than assumed.

**Multiple comparisons, stated in advance.** Three families against a 2σ bar is a
familywise false-positive rate of roughly **1 − 0.954³ ≈ 13%** under the null —
and it would have been ~31% at eight. No correction is applied, because the
sweep is exploratory and a Bonferroni bar would make it unable to detect anything
real at n=3; the number is recorded so "one family passed" is read as what it is.

**A null is ambiguous and does not mean "carries no signal".** Leave-one-out
measures whether a branch is *uniquely necessary*. Two branches carrying the same
signal both show ≈0 delta, and the periodogram pair against the flux family is
exactly that shape. **A null reads as "not uniquely necessary" only**; concluding
"this branch is useless" from it would need a joint drop, which is not in this
sweep.

**The capacity confound.** Every drop removes 7–14% of parameters, so a hurt
could be capacity. The capacity arm bounds it — **+19% params moved nothing**
(paired d = −0.44) — which makes capacity a weak explanation for a large hurt but
not a zero one, and it cuts the wrong way for a *null*: a branch that carries
signal could show ≈0 because the capacity it freed was reused.

**Control:** `models/cv/branches-20260808-rebaseline`, identical shards, identical
seed and splits. Runs go to `models/cv/branches-20260809-drop-{family}`. Nothing
promotes; the registry is untouched.

**One arm spanned a hibernation — recorded before either result is read.** The
laptop's battery reached 1% and the machine hibernated from ~09:17 to 11:00 on
2026-08-09 (`pmset`: `Wake from Hibernate ... Using AC (Charge:1%)`; the process
showed 2 h 14 m elapsed against 59 m of CPU). The processes survived and resumed
correctly, and the damage is bounded:

| arm | status |
|---|---|
| `unfolded` | finished 08:48, ~30 min before the hibernation — **clean** |
| `periodogram` | fold 0 clean; **folds 1–4 span the hibernation** |
| `scalar_only` | started after the wake — **clean** |

Hibernation restores process memory exactly, and TF/Metal would be far more
likely to error than to corrupt silently, so this is probably immaterial —
**probably is not a standard this project accepts for a pre-registered
comparison.** `periodogram` is therefore re-run into
`branches-20260809-drop-periodogram-clean`, and **the clean run is the
authoritative one**, fixed here before either number is looked at.

The hibernation-spanning run is **kept, not deleted**, and the two are compared
as a free consistency check. If they differ by more than the floor, that is a
finding about running training across a suspend — worth having, and it is only
available because the first run was not quietly thrown away.

#### 3.5d Sweep result — every arm null, and the one "PASS" is an artefact (2026-08-09)

Four runs against `branches-20260808-rebaseline`, read by the rule fixed before
any of them existed. TESS slice, each run's own `per_mission.TESS` (n=2,399).

| arm | AUC | ΔAUC | bar | | R@1% FPR | ΔR | bar | |
|---|---:|---:|---:|---|---:|---:|---:|---|
| **control** | 0.9202 | | | | 0.2196 | | | |
| `unfolded` | 0.9135 | −0.0067 | 0.0137 | null | 0.2204 | +0.0008 | 0.0423 | null |
| `periodogram` *(hibernation)* | 0.9215 | +0.0013 | 0.0163 | null | 0.2694 | +0.0498 | 0.0509 | null |
| **`periodogram-clean`** | 0.9213 | +0.0011 | 0.0130 | null | 0.2536 | +0.0340 | 0.0339 | *PASS* |
| `scalar_only` | 0.9094 | −0.0108 | 0.0176 | null | 0.2792 | +0.0596 | 0.0768 | null |

**The `periodogram-clean` PASS clears its bar by 0.00008 — 0.23% of the bar — and
should not be treated as a result.** It satisfies the letter of the pre-registered
rule and is reported as such rather than quietly reclassified, but the
classification is driven by the *bar*, not by the effect:

| run | `pooled_gate_recall_seed_sd` | the three draws |
|---|---:|---|
| control | 0.0292 | 0.1857, 0.1842, 0.2355 |
| `unfolded` | 0.0221 | 0.1162, 0.0838, 0.1260 |
| **`periodogram-clean`** | **0.0029** | 0.1623, 0.1630, 0.1577 |
| `scalar_only` | 0.0597 | 0.1494, 0.2551, 0.1540 |

`periodogram-clean`'s three draws happened to land within 0.005 of each other, an
sd **ten times** the control's tightness — and an sd from three draws carries
~40% sampling spread, so this is far more likely three lucky draws than a
genuinely stabler configuration.

**The comparison that settles it:** `scalar_only` has the **largest** recall
delta in the sweep (+0.0596) and reads *null*, while `periodogram-clean` at
+0.0340 reads *PASS* — purely because one run's 3-draw sd came out 20× the
other's. Under this sweep's own machinery, PASS and null are being assigned by
noise in the error bar rather than by effect size. **Treat all four arms as
null**, and note that the pre-registered consequence of a PASS was never a
decision anyway: it earns a confirmatory repeat.

**What the sweep does support.** `unfolded` — the branch stage 7 exists for, and
the one just rebuilt — costs −0.0067 AUC and +0.0008 recall when removed. Per the
pre-registration that is **not uniquely necessary**, *not* "carries no signal";
redundancy with the flux family is the obvious candidate and a leave-one-out
design structurally cannot separate it. A joint drop would, and is not in this
sweep.

**One pattern, named as a hypothesis and nothing more.** Every drop *raised*
shortlist recall while holding or lowering AUC: +0.0008, +0.0340, +0.0596. All
null individually, but three independent arms agreeing in sign is the same shape
as the capacity arm's 0.145 → 0.236 — which stage 6 showed was a real effect on a
retired architecture. Removing capacity appears to trade ranking for shortlist
recall. **Three nulls pointing one way is not a result**, and this exploratory
sweep cannot establish it; it is recorded as a design question for a later
confirmatory run.

**The hibernation left no measurable trace — and this is only knowable because
the affected run was kept.** The two `periodogram` runs differ by **+0.0158** in
recall against a two-run bar of 0.0383, and by **+0.0002** in AUC against 0.0183:
they agree within noise on both. No evidence that suspending training across a
hibernate perturbs it. The clean run remains authoritative as pre-registered.

#### 3.5e Sequencing: should stage 8 come first? — assessed 2026-08-09

**The question, put by Ollie and never assessed.** Stage 8 invalidates stage 7's
attribution numbers — this file says so twice, and it is why stage 8 was moved
ahead of stage 9. So there is a case for doing stage 8 before finishing stage 7,
on the same "do not pay for it twice" argument that rejected pulling stage 11's
serving parity forward.

**Answer: no. Build stage 7's harness now; defer stage 7's remaining
*attribution compute* until after stage 8.** The question conflates two things
stage 7 still owes, and they sit on opposite sides of the argument.

**1. What stage 8 invalidates is numbers, not code — and the numbers at risk are
four nulls.** Stage 7's 15–20 h estimate was dominated by the sweep, and ~6 h of
it has run. Every arm came back null. Deferring the harness to avoid re-measuring
that protects nothing worth protecting.

**2. The harness is stage 8's instrument, not only stage 7's — this is
decisive.** Stage 8's own success criterion is whether baseline dependence falls,
and the clean test of that was pre-registered on 2026-08-06 as pre-commitment (d):
**injection-recovery on matched hosts with observation baseline held constant.**
That is this harness, including the baseline matcher. Doing stage 8 first means
building stage 8's measuring instrument under stage 8's own pressure — the exact
shape of the argument that rejected pulling serving parity forward, running in
the same direction.

**3. Stage 8 needs a before-reading, and this is the only protocol that can
produce one.** The live 26.4% is already recorded as not comparable to anything
the branch line produces. Without the harness in place first, the pre-stage-8
control-arm rate exists only retrospectively. It is *recoverable* — the fold
checkpoints, the bundles and the labels snapshot are all on disk — so this is
insurance rather than a hard block; but it depends on the pre-stage-8 host
population still being reconstructable after `labels.parquet` changes, and on
someone choosing to go back for it once newer numbers exist.

**4. The cost asymmetry is large, and it is measured rather than estimated.** Two
things expected to be missing are already present, and each removes a chunk of
the build:

| needed | status |
|---|---|
| per-fold scalar normalisation constants | **persisted** in `fold_k/cnn_calibrator.joblib` beside the Platt fit |
| out-of-fold routing for a host | **`predictions.parquet` carries `fold`** |

So the remaining build is the driver, the baseline matcher and the report, over a
chain of existing parts, with compute that is forward passes over 6,721 cached
FITS. Against that, stage 8 is 25–35 h at explicitly **low** confidence, gated on
external catalogue ingestion whose only precedent here (the DV download) was 5×
out.

**What is genuinely paid for twice, and the consequence.** The branch-model
control-arm *reading* itself: stage 8 retrains, so the number moves. It is
therefore recorded as a **baseline, not a verdict**, and the consequence is a
scoping recommendation rather than a silent choice — **no further stage-7 CV
compute before stage 8.** No joint drop, no widening to the five unmeasured
families. That work is the part stage 8 really does invalidate, and the previous
session's own "the natural follow-up is a joint drop" should wait behind it.
**Re-scoping is Ollie's call, so this is put as a recommendation and not acted
on.**
