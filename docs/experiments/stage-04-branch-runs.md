> Moved verbatim from `docs/roadmap.md` §3.2 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.2 Stage 4 — per-diagnostic branches: three runs and a capacity arm

**Stage closed: every arm REJECTED**, all of them on shortlist recall.

#### 3.2a Run 1 — REJECTED (2026-08-05)

**The stop condition fired.** Ollie's third pre-committed case, recorded before
the run: the branch model's all-mission gap is **+0.0222 in the incumbent's
favour on the 4,605 rows both models score**, against a fold standard deviation
of ~0.006. **It beats the incumbent nowhere.** `ca906040` stays served; nothing
was promoted and the registry is untouched.

**The mechanism, measured rather than assumed.** The loss is not spread evenly:

| slice | n | incumbent | branches | gap |
|---|---:|---:|---:|---:|
| TESS | 2,367 | 0.9100 | 0.9079 | **+0.0021** (level) |
| Kepler | 2,238 | 0.9914 | 0.9566 | **+0.0348** |

Almost the whole deficit is Kepler, and the Kepler gap is **monotonic in
transits caught**, by quartile of that count:

| quartile | 1 (fewest) | 2 | 3 | 4 (most) |
|---|---:|---:|---:|---:|
| Kepler gap | +0.0245 | +0.0335 | +0.0416 | **+0.0944** |

TESS is flat on the same split (+0.0002 per quartile step against Kepler's
+0.0218), and against observation *baseline* rather than transit count the
Kepler trend disappears.

**But "TESS is flat" is the wrong lesson, and the quartile split is what makes
it look right.** Quartiles are cut per mission, so TESS's top quartile is a
median of 89 transits where Kepler's is 1,035 — TESS looks immune largely
because it never reaches the regime. Cutting both missions on the *same
absolute* bands (measured 2026-08-06):

| transits caught | Kepler gap | TESS gap |
|---|---:|---:|
| 0–10 | +0.0269 (n=101) | +0.0109 (n=557) |
| 10–30 | +0.0253 (n=206) | −0.0091 (n=873) |
| 30–100 | +0.0243 (n=532) | +0.0034 (n=680) |
| 100–300 | +0.0390 (n=629) | +0.0024 (n=195) |
| **300+** | **+0.0690** (n=770) | **+0.0866** (n=62) |

**The deficit is a function of transits caught, not of mission.** Where TESS
does reach 300+ transits it shows the largest gap in the table — larger than
Kepler's, though on only 62 rows, so treat the point estimate loosely.

**The sharpest form: it is an interaction, not one variable.** Sorting by how
many of the 301 bins the transit itself spans (`duration / period × 301`), the
gap grows *with* span — the opposite of a "transit too narrow to resolve"
story. But span and count correlate (ρ 0.44), and crossing them separates the
two (Kepler):

| transit span | transits caught | n | incumbent | branches | gap |
|---|---|---:|---:|---:|---:|
| <4 bins | <100 | 690 | 0.9835 | 0.9631 | +0.0204 |
| **<4 bins** | **100+** | **149** | **0.9852** | **0.8406** | **+0.1446** |
| 4–8 bins | <100 | 117 | 0.9952 | 0.9639 | +0.0313 |
| 4–8 bins | 100+ | 413 | 0.9942 | 0.9399 | +0.0543 |
| 8+ bins | 100+ | 837 | 0.9915 | 0.9317 | +0.0597 |

Within *every* span band the gap still tracks transit count, and the worst cell
by a factor of three is **narrow in phase and caught many times**: 149 Kepler
targets with a median transit spanning **3.1 of 301 bins** and a median of
**134 transits caught**, where the incumbent scores 0.9852 and the 301-bin model
0.8406. At 2001 bins those same transits span ~21 bins. TESS shows the same
shape where it reaches the regime: its 8+ span / 100+ count cell (n=207) is
+0.0425, against −0.0086 and −0.0059 in the two wider-span low-count cells.

*(Corrected 2026-08-07 by regeneration from the artefacts. This previously read
"every low-count TESS cell sits at or below zero", which is false: the narrow-span
low-count cell is **+0.0243 on n=417**, the largest of the three. It weakens the
interaction reading — on TESS a narrow span alone carries a gap without a high
transit count — though run 2 falsified the resolution hypothesis regardless.)*

That is a coherent resolution mechanism rather than a hand-wave: many folded
transits make the per-bin median precise enough for fine structure to exist in
the data, and a coarse grid then smears it away — worst where the feature is
narrowest in phase.

**Pre-registered consequence.** If resolution is the cause, the **+0.1446 cell
improves most** in run 2, and the TESS 300+ band improves too. If the gap is
flat across these cells after the change, the effect is something other than
resolution and the reading below should be treated as falsified regardless of
what the headline Kepler number does.

**Both facts stand, and the second does not undo the first.** The mechanism
refines *why* the model lost; it does not un-fire the stop condition. The
resolution restoration below is a **new registered experiment with its own
pre-registered reading**, not a rescue of this run. Run 1 is rejected and stays
rejected whatever the resolution run returns.

Also not like-for-like, and recorded so the comparison is not overread: the
incumbent is Optuna-tuned and trained with augmentation on 2001/201 views; run 1
was a first pass at 193k params, two conv blocks, **no augmentation and no
tuning**. Augmentation now exists for the view set (`viewset_augment.py`) and is
declared in `conf/model/cnn_branches.yaml` at the same magnitudes the incumbent
trained with. Tuning stays out of run 2 deliberately — it would confound the
resolution test, and it is the Optuna step at the end of the old stage 2 — now
stage 10.

#### 3.2b Pre-commitments recorded before the next result exists

Written down first so they cannot be adjusted to fit an outcome.

*(Not renumbered — this block is the verbatim record of what was committed to on
2026-08-06. New stage numbers appear in square brackets where a reference would
otherwise be unresolvable; the mapping table is at the top of this file.)*

**(a) The capacity re-run is CANCELLED, not deferred.** Recorded prediction:
`init_filters=22` — 226,711 parameters, 0.4% from the incumbent's 227,641 —
would move the Kepler gap by **less than 0.005**. Capacity is not the binding
constraint at a 0.0348 gap.

**MANDATORY TRIGGER, recorded with it:** if the resolution fix fails to close
the Kepler gap, the capacity run becomes **obligatory**, not optional.
Cancelling a pre-registered run after an unfavourable result is a red flag by
construction, and this guard is the whole reason the claim stays falsifiable.
The guard is the commitment; the cancellation is only its consequence.

Note also that the resolution change moves the parameter count on its own, so it
**partially subsumes the capacity question** — a confound to state rather than
to claim as a bonus.

#### 3.2c The trigger, re-derived against run 3 — recorded 2026-08-08, before run 3 was read

The trigger fired legitimately and is **not** being cancelled. Its antecedent is
true even under the corrected comparison: re-baselined, run 2's Kepler gap is
**+0.0685** against a ~0.020 falsification threshold, and the re-baseline barely
moved the incumbent (0.9914 → 0.9915). The resolution hypothesis is properly
falsified.

What changed is not the trigger's validity but **its target**. `init_filters=22`
was specified against run 1's architecture, which carried the missingness
mission-indicator, two towers emitting `relu(bias)` on 56% of rows, a dead
`bootstrap_significance` and mission-blocked batches. None of those exist now.
Running it as written would answer a question about a model that has been
retired. So the obligation is discharged by asking the capacity question of the
*current* architecture, under a rule written before the result is known:

| run 3's Kepler gap | consequence |
|---|---|
| under **~0.012**, TESS not regressed | capacity is **not** the binding constraint. The arm is **redundant, not cancelled** — a 215,281-parameter model matched a 227,641-parameter incumbent, which is a stronger answer than the capacity run could have produced |
| **~0.020 or worse** | the capacity arm is **mandatory**, run as `init_filters=22` on run 3's architecture |
| in between | report as-is and decide in writing. **No default** |

The arm is specified as within-architecture (run 3 at `init_filters=16` vs `22`)
because cross-architecture parameter counts are not a clean capacity control —
the incumbent is a 9-dim dual-view CNN, not a branch model, so "215,281 below
227,641" is suggestive and not conclusive on its own.

**TESS still gates.** A Kepler reading cannot promote anything on its own, and
recall @1% FPR is read alongside AUC: run 2 sat within 0.016 of the incumbent on
TESS AUC while catching **less than half** as many real planets at the shortlist
threshold.

**(b) Gate population — three tiers.**

| tier | slices | role |
|---|---|---|
| **gates** | TESS | 100% of the deployment population |
| **mandatory diagnostic, alarmed** | Kepler, K2 | a >0.02 AUC drop does not block promotion but **requires a written explanation in the roadmap first** |
| **reported, never gates** | all-mission | its weights are a sampling artefact |

The aggregate is unfit to decide anything: **Kepler is drawn at exactly
1,250/1,250 by construction**, so any all-mission number is weighted by a
sampling decision, and the 4,605-row comparison weighted **Kepler 48.6% / TESS
51.4% / K2 0%** in a decision whose real consequences are 100% TESS.

**This is not goalpost-moving, and here is the check that shows it:** on TESS
alone the branch model **still fails**, on recall @1% FPR — 0.238 against the
incumbent's 0.307. Narrowing the gate to the deployment slice does not rescue
run 1.

**Scope:** training and evaluation populations are **unchanged** — all three
missions, always, reported per mission. Only the promote/reject *decision rule*
narrows. Enforced in `validation/promotion.py`; summaries without a
`per_mission` block fall back to pooled means.

**(c) Recall @ 1% FPR is now a first-class gate criterion**, alongside AUC and
Brier. AUC scores ranking at every threshold; a follow-up shortlist lives at
exactly one. Current TESS numbers:

| | incumbent | branches |
|---|---:|---:|
| recall @1% FPR | **0.307** | **0.238** |
| recall @5% FPR | 0.561 | 0.550 |
| recall @10% FPR | 0.731 | 0.689 |
| Brier | 0.1211 | **0.1194** |

The branch model is better *calibrated* on TESS and worse where it is used.

**(d) Queued, not built: injection-recovery on matched hosts with observation
baseline held CONSTANT.** It is the only causal measure of detection
performance available here, and it is immune to the label-selection confound
that moved to stage 3 [now 8]. Build it when 2(b) [now stage 7] is actually run.

#### 3.2d K2 was unbenchmarked for 9.7% of training — now it is not

The incumbent's `predictions.parquet` holds 4,818 out-of-fold rows and **zero
K2**: that run predates K2 in the catalogue. Every comparison against it
inner-joined all 527 K2 rows away silently.

Scored 2026-08-06 (`pipeline/scripts/evaluate.py score`,
`results/incumbent_k2_benchmark.json`):

| K2, n=527 | incumbent | branches |
|---|---:|---:|
| ROC-AUC | **0.9348** | 0.9189 |
| Brier | 0.1538 | **0.0957** |
| ECE | 0.1989 | **0.0500** |
| recall @1% FPR | **0.190** | 0.089 |

**Read it with its asymmetry.** No K2 row was in any of the incumbent's
training folds, so its numbers are **zero-shot cross-mission transfer**; the
branch model's are ordinary out-of-fold with K2 in four folds of five. Ranking
is comparable and the incumbent wins it by 0.0159 despite the handicap.
Calibration is not comparable — the incumbent's Platt scalers were fitted on
Kepler+TESS validation rows and K2's base rate is 0.598, which is most of the
0.1989 ECE.

Two rebuild details that would otherwise return confident wrong numbers: the
9-dim and 13-dim aux layouts **disagree at index 7** (catalogue SNR vs
`pink_snr`), so the vector is rebuilt rather than sliced; and catalogue SNR is
absent on **all 527** K2 rows, so that lane imputes.

`eval/comparison.py` now reports per-mission coverage whenever two prediction
sets are compared, and names any mission an inner join drops entirely. A mission
falling out of a comparison cannot be silent again.

#### 3.2e Run 2 — the resolution fix, pre-registered 2026-08-06

**One change, both halves together: global 301 → 2001, local 31 → 201.** They
are the same mechanism testing the same hypothesis; splitting them costs two
runs to answer one question. Global-vs-local attribution is a follow-up
ablation and only matters if the fix works. Everything else is held:
`init_filters=16`, `conv_blocks=2`, same folds, same seed, plus the augmentation
built for this run.

**Sizing, measured before launch** (296-target probe at the new resolution;
peak RSS sampled during `fit()`, not read off at the end):

| | 301/31 | 2001/201 | |
|---|---:|---:|---|
| parameters | 192,817 | **233,617** | +21.2% |
| shards on disk | 122 MB | **~669 MB** | 126.4 KB/example, ×5.5 |
| interim per-target cache | 92 MB | ~360 MB | |
| peak training RSS | 4,861 MB | **~5.4 GB** | on 26 GB; ~4.7 GB is fixed TF/Metal cost, not dataset size |

**State the confound rather than bank it.** At 233,617 parameters the resolution
change lands *above* the incumbent's 227,641, and above the 226,711 the
cancelled capacity run (`init_filters=22`) would have reached. So this run
carries more capacity than the capacity experiment would have, and **a Kepler
gain cannot be attributed to resolution alone**. That is the sense in which it
partially subsumes the cancelled run — and it is a reason the pre-registered
reading below is about the *size* of the move, not its existence.

**How the result will be read** — fold std ~0.006, committed before the run
finishes:

| outcome | reading |
|---|---|
| Kepler gap closes to **under ~0.012** and TESS does not regress | resolution was the cause. Proceed to 2(b) [now stage 7] |
| Kepler gap roughly **halves** | plausible, unproven. Report as-is, proceed |
| Kepler gap stays **above ~0.020** | resolution hypothesis **FALSIFIED**. The capacity run becomes mandatory under the trigger above. Say so and **stop; do not tune** |

**TESS must not regress.** TESS AUC *and* recall @1% FPR within noise of the
current branch model. A Kepler win bought with a TESS loss is a failure, not a
trade. K2 is reported alongside both — 9.7% of training, and benchmarked for
the first time on 2026-08-06.

**Second test, on the gating mission.** The 62 TESS targets with 300+ transits
caught carry a +0.0866 gap — the largest in the absolute-band table above. If
resolution is the cause, that band improves too. It is 62 rows, so it cannot
carry a decision on its own; it is recorded as a directional check that the
mechanism is not a Kepler-only story.

#### 3.2f Run 2 result — the resolution hypothesis is FALSIFIED (2026-08-07)

`models/cv/branches-20260807-2001`. Gate: **REJECT**. `ca906040` stays served.

**It made every slice worse, and roughly doubled the gap it was meant to close.**

| slice | incumbent | run 1 (301/31) | run 2 (2001/201) | run 2 gap |
|---|---:|---:|---:|---:|
| TESS *(gates)* | 0.9100 | 0.9079 | **0.8944** | **+0.0156** |
| Kepler | 0.9914 | 0.9566 | **0.9207** | **+0.0707** |
| all | 0.9558 | 0.9337 | **0.9043** | **+0.0516** |
| TESS recall @1% FPR | 0.307 | 0.238 | **0.126** | |

The pre-registered reading said a Kepler gap **above ~0.020 falsifies**. It went
from +0.0348 to **+0.0707**. The sharper cell-level test fails the same way:
Kepler 0–10 transits went from +0.0269 to **+0.2038**, so the damage is
concentrated where evidence is *thinnest* — the opposite of what a resolution
deficit predicts.

**Under the stage 4 trigger the capacity run is now MANDATORY**, and per the
pre-registration this stops here rather than tuning.

*(The trigger stands and the falsification holds — the Kepler gap is +0.0685 even
re-baselined. But `init_filters=22` was specified against run 1's architecture,
which no longer exists. The obligation was re-derived against run 3 on
2026-08-08, before run 3 was read: see "The trigger, re-derived against run 3"
above.)*

#### 3.2g Run 3 result — the fixed architecture on the fixed shards (2026-08-08)

`models/cv/branches-20260807-shared`. Gate: **REJECT**, on shortlist recall.
`ca906040` stays served; the registry is untouched.

The first run of the shared flux tower (215,281 params, *below* the incumbent's
227,641) on shards with the four defects fixed. Neither run 1 nor run 2 is a
baseline for it; the comparison is against the re-baselined incumbent, on the
5,375 rows both score:

| slice | n | incumbent | run 3 | gap | inc R@1% | run 3 R@1% |
|---|---:|---:|---:|---:|---:|---:|
| K2 | 527 | 0.9348 | 0.9028 | +0.0320 | 0.191 | 0.137 |
| Kepler | 2,481 | 0.9915 | 0.9464 | +0.0451 | 0.829 | 0.363 |
| **TESS** *(gates)* | 2,367 | 0.9100 | **0.9130** | **−0.0030** | **0.307** | **0.145** |
| all | 5,375 | 0.9523 | 0.9251 | +0.0272 | 0.439 | 0.263 |

**On TESS AUC it is the first branch model to reach the incumbent** — run 1
+0.0021, run 2 +0.0156, run 3 −0.0030 — and it is better calibrated on the
gating slice (Brier 0.1150 vs 0.1211, ECE 0.0171 vs 0.0438).

**And it is rejected anyway, on the criterion that matters.** TESS recall @1%
FPR is **0.145 against 0.307**: at the shortlist threshold it catches *less than
half* as many real planets. That is the same failure as runs 1 and 2 (0.238,
0.126) and it is what pre-commitment (c) exists to catch — a model that ranks
comparably overall and is worse exactly where it is used.

**The −0.0030 TESS win is not a win.** See the variance decomposition below: a
margin under ~0.009 is inside the noise. Level is the honest reading.

#### 3.2h The variance decomposition, measured for the first time

`--n-models-per-fold 3` makes `summary.variance` report the two components apart:

```
seed_sd 0.0081   fold_sd 0.0094   n_models_per_fold 3
```

- **`seed_sd` = 0.0081** is per-*model* training noise. It independently
  confirms the 0.0106 floor measured by re-running fold 0 five times in five
  processes — a different measurement path, arriving slightly lower.
- **`fold_sd` = 0.0094** is fold difficulty, a property of the split.
- They are **nearly equal**, so the ±0.0106 this project has been quoting was
  never "the run's uncertainty": it was roughly `sqrt(seed² + fold²)` with the
  two halves indistinguishable.
- Run 3 averages 3 models per fold, so the run-level reseeding sd is about
  `seed_sd/√3 ≈ 0.0047`. **A margin under ~0.009 is not a decision.**

#### 3.2i The capacity arm — trigger fired, launched 2026-08-08

Kepler is **+0.0451**, well past the ~0.020 threshold in the re-derived trigger
above, so the arm is **mandatory** and running as
`models/cv/branches-20260808-capacity` (`--init-filters 22`).

**One correction to the pre-registration, recorded rather than absorbed.** The
original prediction called `init_filters=22` "226,711 parameters, 0.4% from the
incumbent's 227,641" — computed on the *four-tower* architecture. On the shared
tower it is **256,711**, so this is a **+19% capacity test rather than the
incumbent-parity test it was designed as**. That makes the falsification
stronger, not weaker: if 19% more capacity does not close a +0.045 Kepler gap,
capacity is not the binding constraint.

#### 3.2j Capacity arm result — capacity is NOT the constraint (2026-08-08)

`models/cv/branches-20260808-capacity`, `init_filters=22`, 256,711 params
(+19% on run 3). Gate: **REJECT**. `ca906040` stays served.

**The clean test, paired fold by fold against run 3** — same split, same seed,
same shards, so unlike run 1 vs run 2 the pairing here is exact:

```
mean -0.0035, won 3/5, d=-0.44
```

Against a run-level reseeding sd of ~0.0047, that is **nothing**. On the shared
rows, +19% capacity moved the Kepler gap the wrong way:

| slice | incumbent | run 3 (16f, 215k) | capacity (22f, 257k) |
|---|---:|---:|---:|
| **TESS** *(gates)* | 0.9100 | **0.9130** | 0.9089 |
| Kepler | 0.9915 | 0.9464 (+0.0451) | 0.9449 (**+0.0466**) |
| K2 | 0.9348 | 0.9028 | 0.8997 |
| TESS recall @1% FPR | **0.307** | 0.145 | **0.236** |

**The trigger is discharged, and the cancelled prediction was right.** The
2026-08-06 pre-commitment recorded that `init_filters=22` "would move the Kepler
gap by less than 0.005". Measured: **−0.0015, in the wrong direction.** The
difference is that it is now tested rather than asserted, which is exactly what
the trigger existed to force. **Capacity is closed as a hypothesis.**

**One observation, deliberately not upgraded to a finding.** TESS recall @1% FPR
went **0.145 → 0.236** while AUC fell — capacity traded ranking for shortlist
performance. That is a large relative move on the criterion that has rejected
every run so far, but it is one run on a single-threshold statistic with no
variance estimate, and this project has been burned by exactly that shape of
evidence. It is worth a `--n-models-per-fold` repeat aimed at recall
specifically before anyone builds on it.

Also of note: `fold_sd` rose 0.0094 → 0.0147 while `seed_sd` held at 0.0082.
The extra capacity made folds diverge without making individual draws noisier.

#### 3.2k Three training-path changes that break comparability going forward

Recorded here because this project's recurring injury is a comparison that is
not like-for-like and does not say so.

**Augmentation masking is now gated by view kind (2026-08-08).** It previously
applied to all eleven views. Zero is the out-of-transit baseline on folded flux,
so masking there removes a measurement — but zero in `gap_view` asserts *no
cadence was missing*, and zero in a peak-normalised periodogram asserts *no
power at this period*, both while the presence channel still reports the bin as
measured. Three of eleven views were being fed a confident false claim on
`mask_prob` of their bins every epoch.

**Run 3 and the capacity arm are unaffected** — both had the module loaded
before the change — so the comparison that decides the capacity question is
still internally consistent. **Every run after them is not comparable to run 3
on this axis** and needs its own baseline. Stage 6 produces exactly that
re-baseline, and stage 7 re-baselines again anyway.

**The unfolded branch was rebuilt (2026-08-08).** Audit finding #23: it
convolved along the transit axis with the 201 phase bins flattened into 603
unordered channels, so it never saw a transit. It now runs a per-transit conv
tower under `TimeDistributed` and pools with a masked mean + max + spread. The
model drops **215,281 → 169,361 parameters (−21.3%)**, almost all of it the
48,256-parameter convolution that was doing the damage.

Runs 1, 2, 3 and the capacity arm all carried the broken branch, so **stage 4's
rejections stand** — nothing about this changes what those runs measured,
and a branch that could not see a transit is one more reason they lost. But no
run so far is a baseline for the rebuilt model. The direction is the opposite of
the capacity arm's (+19%, paired d = −0.44, nothing), which is weak evidence
that −21.3% is not decisive on its own. Weak evidence, not a measurement.

**The inner validation split is now stratified (2026-08-08).** It was
`GroupShuffleSplit`, which is group-aware but guarantees nothing about class
balance — and the Platt fit downstream requires both classes, or it converges
happily on a scaler that maps every score to one end. Both affected sites now
call `training/splits.py`'s `stratified_inner_split`; the random-forest holdout
at `train.py:132` is a test split and is untouched.

This changes the inner partition and therefore the numbers, which is why it had
been deferred to "between experiments". It landed here because the unfolded
rebuild already forces a fresh baseline — **one re-baseline absorbs both changes
instead of two**. Production was never at risk (~868 validation rows); it was
the tiny test fixtures that exposed it.

#### 3.2l The one cell three architectures have not moved

The narrow-span, high-count Kepler cell is unchanged across every run:

| | run 1 (301/31) | run 3 (shared, 2001/201) |
|---|---:|---:|
| span <4 bins, 100+ transits (n≈150) | **+0.1446** | **+0.1448** |

Two bin resolutions, four fixed input defects, tied odd/even weights and a
shared tower move it by **0.0002**. Whatever drives that cell is not resolution
(run 2 made it worse), not the missingness indicator, and not the separate
towers. It is the sharpest unexplained thing in the model and it deserves its
own investigation rather than another architecture pass.

#### 3.2m What the run also uncovered: the noise floor was never measured

Fold 0 was re-run five times through `run_fold` with the trainer's own seeding,
one process each:

```
0.8927  0.8942  0.8984  0.9083  0.9179
mean 0.9023   sd 0.0106   range 0.0252
```

`set_global_seed`'s docstring already says it "doesn't make TF fully
deterministic on GPU", and nothing sets `enable_op_determinism` — so this is
known behaviour that had simply never been quantified. **Single-fold training
noise is sd ≈ 0.011.**

Two things follow, and they point in opposite directions:

- **It does not overturn the result.** The 5-fold mean averages this down —
  between 0.0048 (independent folds) and 0.0106 (fully correlated). Run 1 vs
  run 2 differ by 0.0313, which is **2.9σ to 6.6σ**. The falsification stands
  comfortably.
- **But the `±` in every `cv_summary.json` is not the run's uncertainty.** It is
  the spread *across folds within one run*, which mixes genuine fold-to-fold
  variation with training noise and says nothing about whether re-running the
  same configuration would reproduce the headline. Any future decision on a
  margin under ~0.02 needs repeat runs, not a single run's fold std.

  *(The ~0.02 was a conservative guess made while the two variance components
  were still inseparable. **Superseded 2026-08-08 by the measured figure: a
  margin under ~0.009 is not a decision** — see "The variance decomposition,
  measured for the first time" above, where `--n-models-per-fold 3` finally
  split `seed_sd 0.0081` from `fold_sd 0.0094`. That threshold is **AUC only**;
  recall @1% FPR still has no variance estimate, which is precisely why the
  capacity arm's recall jump cannot be acted on as it stands. **That gap is
  stage 6**, below.)*

**One knock-on, deferred deliberately.** The candidate view set
(`data/processed/candidates_viewset/`, 5,347 rows) is still at 301/31, so a
run-2 model cannot score candidates until it is rebuilt — about two hours. That
blocks the candidate-population bias measurement and stage 7's control arm,
but not run 2's own promote/reject decision, and it is wasted work if the
resolution hypothesis is falsified.

*(Superseded 2026-08-08: the two-hour figure assumed a cold cache and was then
overtaken twice. `data/interim/viewset/g2001l201` had accumulated **5,426
targets, 309 MB** at 2001/201, which would have made the rebuild mostly cache
hits — but the `_cache_path` ephemeris key renames every entry, so none of them
can be found and the build re-derives from the light curves. Budget a cold
rebuild of all 5,347 candidates.

Two orphaned caches, for two different reasons — measured 2026-08-08, do not
conflate them:

| path | targets | size | resolution | orphaned by |
|---|---:|---:|---|---|
| `data/interim/viewset/g2001l201/` | 5,426 | 309 MB | 2001/201 | the `_cache_path` ephemeris key — **a cost this change introduced** |
| `data/interim/viewset/*.npz` (loose) | 5,423 | 92 MB | 301/31 | superseded resolution, already dead before the key change |

401 MB total is reclaimable, but only the 309 MB is attributable to the
ephemeris key. Both are under the old `{mission}_{tic}.npz` naming and nothing
reads either.)*

**Both deleted 2026-08-08: 10,849 files, 395.9 MB** (312.5 + 83.4; the 401 MB
above was MB-vs-MiB, not a different set). `data/interim/` is gitignored and
carries no DVC pointer, so this is derived data with no artefact behind it.

**The delete had a trap in it worth recording.** By then `g2001l201/` held *two*
generations side by side — the 5,426 orphaned training-target entries under
`{mission}_{tic}.npz`, and the **5,346 ephemeris-keyed candidate entries the
rebuild had just written**, 326.8 MB of them. A glob over the directory would
have taken both. The two were separated by regex, asserted to be disjoint before
anything was unlinked, and `_cache_path` was then re-run over 3,000 catalogue
rows to confirm the survivors are still addressable: **2,995 hits, 5 misses.**

Note the asymmetry this leaves: what survives covers **candidates**, not
training targets. A future *training* view set rebuild is still cold — it was
already, since those entries were unfindable — so the cache on disk now helps a
candidate re-run and not a training one.
