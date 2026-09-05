> Recorded 2026-09-05 03:52, while arm D'' was still training and no contrast
> had been computed. Kept verbatim as the pre-registration the
> [result](phase-1-result.md) is read against. Frozen.

# How Phase 1 will be read — fixed before the result existed

Recorded before the numbers exist, per the roadmap's pre-registration rule. Arm
C'' finished 01:38; arm D'' is still training, so no contrast has been computed.

## The pre-registered statistic

§4.2c Phase 1: "this is a **mechanism test** of finding 2, not a performance
bid. If TESS recall @1% FPR on `dv_usable` rows does not move beyond its floor,
the reference-frame explanation is **falsified** and is to be recorded as such —
the branch is then simply not carrying signal, and no third stamp variant is
commissioned."

So: **D'' − C'' on TESS recall @1% FPR, restricted to `dv_usable` rows.**
One number. Everything else on the page is secondary and reported as such.

## Which floor, decided now rather than after

Two floors exist and they are not the same measurement:

1. **The member-pairing floor** the analysis script computes for the paired
   contrast, marginalised over all 3! = 6 member pairings, max as the bar. This
   is stage 9's method, and the script was verified against stage 9's published
   0.0843 and 0.0740 before it was used here.
2. **The Phase 1a seed floor**, 0.0432 on `dv_usable`, which §4.2c names when it
   says "Floors: from the Phase 1a seed sweep below".

The pairing floor is the right instrument for a *paired* contrast — it is the
spread of exactly the comparison being made — and the seed floor is the right
one for a *cross-run* comparison. Phase 1 is paired: the drop is applied at the
model, both arms consume identical augmented batches.

**Decision, taken before the result:** the pairing floor is the bar, because it
matches the design. The Phase 1a seed floor is reported beside it as a
cross-check. **If the two disagree on the verdict, the result is recorded as
UNRESOLVED and neither reading is banked** — that is a genuine ambiguity in the
pre-registration, and choosing whichever floor gives the tidier answer after
seeing the numbers is exactly what this project's rules forbid.

## What each outcome means

| outcome | what is recorded |
|---|---|
| margin beyond the bar, positive | the reference-frame explanation survives its first real test. Not a promotion: the branch line's replacement rejections stand |
| margin inside the bar | **the reference-frame explanation is FALSIFIED.** The branch is not carrying signal, no third stamp variant is commissioned, and the branch line closes in writing |
| the two floors disagree | UNRESOLVED, and Ollie decides which floor governs before anything is banked |

## What is NOT read from this

- Nothing promotes. `models/registry.json` stays untouched either way.
- TESS recall on **all** rows is secondary: 58.9% of rows have the difference
  branch gated off by construction, so an unstratified reading is diluted below
  the floor before it starts. That trap cost stage 9 its primary criterion.
- ROC-AUC is reported and does not decide. The stage exists to test a mechanism
  behind shortlist recall.
