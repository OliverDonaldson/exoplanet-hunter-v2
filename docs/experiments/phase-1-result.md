> Written 2026-09-05, after both arms landed. The reading criterion below was
> fixed at 03:52, before arm D'' finished at 04:50 and before any contrast was
> computed. Frozen on the same rule as every other file here: a correction is a
> dated note appended under the entry it corrects, never an edit.

# Phase 1 — result: the reference-frame explanation is FALSIFIED (2026-09-05)

Arm C'' 1 h 58 (2026-09-04 23:40 to 2026-09-05 01:38), arm D'' 3 h 12 (01:38 to
04:50), both at `--n-models-per-fold 3` on `models/fold_assignments/stage10_5.json`,
both from code pinned at `d93a1a0` in a detached worktree, both over the Phase 1
shard set built by `build_viewset.py --views-from`. Seed 44's control-arm pass
ran alongside and completed Phase 1a.

## What was being tested

[4.2c](phases-pre-registration-4-2c.md) specified Phase 1 as a **mechanism test
of finding 2** of the ExoMiner++ readout, not a performance bid. Finding 2 held
that stage 9's difference-image branch read null because its stamps were fed
without the star's own pixel position, so the network had no reference frame in
which a centroid shift means anything. Phase 1 adds that position as a fourth
stamp channel — a bilinear sub-pixel marker at DV's `ticReferenceCentroid` — and
asks whether the branch then carries signal.

The design is the same paired model-level drop stage 9 used. The drop is applied
at the **model**, not the shard set: every `Input` stays in the signature and the
stream yields all fourteen views to both arms, so C'' and D'' consume identical
augmented batches and only the difference branch differs between them.

## The reading criterion, fixed before the result

4.2c: *"If TESS recall @1% FPR on `dv_usable` rows does not move beyond its
floor, the reference-frame explanation is **falsified** and is to be recorded as
such — the branch is then simply not carrying signal, and no third stamp variant
is commissioned."*

Two floors exist and they measure different things, so which one governs was
decided **before** the numbers, at 03:52 while arm D'' was still training and
kept verbatim in [phase-1-reading-criterion.md](phase-1-reading-criterion.md):

- the **member-pairing floor**, marginalised over all 3! = 6 pairings with the
  maximum as the bar — stage 9's method, and the instrument verified against
  stage 9's published 0.0843 and 0.0740 before it was used here;
- the **Phase 1a seed floor**, 0.0432 on `dv_usable`, which 4.2c names when it
  says *"Floors: from the Phase 1a seed sweep"*.

The pairing floor was fixed as the bar because it matches a paired design, with
the seed floor as a cross-check, and it was recorded in advance that **if the
two disagreed the result would be UNRESOLVED and neither reading banked**. They
do not disagree.

## Result

**TESS, `dv_usable` rows — the pre-registered statistic.** n = 2,077, 1,220 positive.

| | arm C'' (branch dropped) | arm D'' (branch kept) | D − C | max floor | x | verdict |
|---|---:|---:|---:|---:|---:|---|
| **recall @1% FPR** | 0.2197 | 0.1943 | **−0.0254** | 0.0979 | **0.26x** | **within floor** |
| ROC-AUC | 0.9228 | 0.9188 | −0.0040 | 0.0154 | 0.26x | within floor |

Against the cross-check floor the margin is 0.0254 against 0.0432 — inside that
one too. Both floors agree, so the pre-registered ambiguity never arose.

**TESS, all rows,** reported and secondary: 58.9% of rows have the branch gated
off by construction, so an unstratified reading is diluted below its floor
before it starts. That trap cost stage 9 its primary criterion.

| | arm C'' | arm D'' | D − C | max floor | x | verdict |
|---|---:|---:|---:|---:|---:|---|
| recall @1% FPR | 0.2208 | 0.2015 | −0.0192 | 0.1266 | 0.15x | within floor |
| ROC-AUC | 0.9176 | 0.9160 | −0.0016 | 0.0203 | 0.08x | within floor |

## Read exactly as pre-registered

**The reference-frame explanation is FALSIFIED.** TESS recall @1% FPR on
`dv_usable` rows did not move beyond its floor — it reached 0.26x of it — so per
4.2c the branch is not carrying signal and **no third stamp variant is
commissioned.**

Two things are worth stating because omitting either would flatter the result.
The margin is **negative in every cell**: adding the target-position channel did
not help, and if anything cost a little. None of those movements is a measured
difference, so the honest statement is *no effect*, not *a small harm*. And the
mechanism this phase tested was the best remaining explanation for stage 9's
null — 4.2b named it as the one input ExoMiner++ feeds that we did not. Feeding
it changed nothing, so the explanation is not merely unsupported, it is spent.

## One post-hoc number, labelled as such

TESS ECE improves 0.0342 → 0.0238 from C'' to D''. No ECE floor was
pre-registered, so this is not a measured difference and is not banked. It is
recorded because it is the only cell where arm D'' leads, and stage 9 recorded
its own ECE movement the same way; suppressing it here would be as selective as
banking it.

## Phase 1a is complete

Seed 44's control-arm pass landed, so all four floors 4.2c asked for are
measured on three draws of one configuration:

| statistic | mean | sd | floor, 2·sd/√3 |
|---|---:|---:|---:|
| TESS ROC-AUC | 0.9162 | 0.0021 | 0.0024 |
| TESS recall @1% FPR | 0.2136 | 0.0185 | 0.0213 |
| TESS recall, `dv_usable` | 0.1861 | 0.0374 | 0.0432 |
| **control-arm host-AUC** | **0.5826** | **0.0172** | **0.0198** |

Host-AUC had never had a floor. It does now: 0.5639, 0.5862, 0.5977 across seeds
44, 43 and 42.

**The RNG limit, read as pre-registered.** Stage 9 offered RNG drift as the
probable cause of arm C's −0.0515 anchor gap. The sweep's own spread is 0.0369,
which does not contain it, so **the RNG explanation does not account for that
gap**. This remains three draws, and 3.11d warns about exactly that; seed 45
would widen it. The pre-registration is reported as written either way.

## Gate

Both arms were run through `promotion_gate.py` with
`--champion-summary models/cv/champion-rebaselined-today/cv_summary.json`,
because the champion's own summary carries no `per_mission` block. **Both
returned UNRESOLVED** (exit code 2), each writing a `promotion_log.json` into
its own run directory. `models/registry.json` is byte-identical before and
after, `sha256 0dee467c351ae363…`, and `ca906040` still serves.

UNRESOLVED rather than REJECT is the substantive reading: arm C'' beats the
champion on TESS ROC-AUC (0.9176 against 0.9100) and on Brier and ECE, while its
shortlist recall of 0.2208 sits 0.0862 below the champion's 0.3069 — **1.1x its
own floor**, which the gate calls too close to resolve from three draws. The
branch line does not replace the champion and does not cleanly lose to it
either. This is the first time 4.1b's third verdict has been reached by a real
run rather than by a test.

## W10 measured inside a single run, for the first time

Fold completion times, in minutes:

| run | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | total |
|---|---:|---:|---:|---:|---:|---:|
| arm C'' | 17 | 27 | 22 | 20 | 31 | 1 h 58 |
| arm D'' | 25 | 23 | 29 | 36 | **78** | 3 h 12 |

Arm D''s last fold took **3.1x** its first, on identical work. W10 was recorded
as a slowdown *across* repeated `run_cv` calls in one process; this is the same
pathology *within* one call, and it is most of the 74-minute gap between the two
arms rather than the one extra branch. The mitigation is unchanged and it held:
one process per CV run.

## Limits carried forward

- Arm C''`s TESS recall is 0.2208 against stage 9 arm C's 0.2315 on the same
  code and the same drop. The shard sets differ by two views, so these are
  independent draws rather than replicates, which is the standing limit
  [stage 9](stage-09-difference-image.md) recorded and this result does not
  close.
- The floors here are three draws. Every conclusion above survives the wider
  reading, but the next result to lean on these quantities should widen them.
- Stage 9's primary criterion, control-arm host-AUC on the difference branch,
  remains unrunnable for the reason 4.2 records: the stage 7i harness zeroes
  every DV input, so the branch contributes exactly 0.0 on control-arm hosts.
  Phase 1 does not change that and did not try to.
