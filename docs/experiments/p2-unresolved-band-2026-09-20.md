# P2 — the UNRESOLVED band's lower edge · pre-registration · 2026-09-20

Written **before any code changes**, per rule 6. Nothing here is adjusted after the
result; a result landing outside these terms is reported as falsified.

This changes the gate's decision rule, which is a larger change than
[#123](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/123) proposed.
**No model is trained and nothing is promoted.**

Issue [#128](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/128),
following the size measurement in
[`p2-gate-size-2026-09-20.md`](p2-gate-size-2026-09-20.md).

## How this will be read, fixed before the work starts

It **succeeds** if the gate's decision rule becomes monotone in the true margin, if
its measured size under a true null falls to single figures, and if its power at the
MDE the project publishes matches what that MDE asserts.

It is **falsified** by any one of:

1. **Measured size under the new rule stays above 5%.** The account in §1 is
   geometric and predicts 2.3% analytically, ~1.6% once the other tolerances are
   applied. If it does not fall that far, the fall-through is not the mechanism and
   the diagnosis is wrong.
2. **Measured power at a true margin of one MDE stays below 70%.** The change is
   justified by internal consistency with the MDE formula; if the rule still does
   not deliver what that formula asserts, the justification fails.
3. **Any recorded verdict moves to PROMOTE on a margin that did not exceed its own
   floor.** Verdicts moving to PROMOTE *because the margin did clear the floor* are
   the rule working and are not a failure — see §5.

## 1. What the rule is now, and why it is broken

`promotion.py:808` REJECTs any `cand_auc <= champ_auc`, so `unresolved_against` is
only ever reached with a positive AUC margin. It marks UNRESOLVED on
`[floor/1.5, floor*1.5]`. With `floor = 2 x se` that is `[1.33 se, 3 se]`, and a
positive margin **below** 1.33 se is outside the band, has not been rejected, and
reaches PROMOTE at `promotion.py:917`.

Three consequences, analytic, normal approximation, floor correctly sized. `d` is
the true margin in units of `se`:

| true margin | PROMOTE — via the hole | via the tail | **total** |
|---|---:|---:|---:|
| 0 — the null | 40.9% | 0.1% | **41.0%** |
| 1.0 se | 47.2% | 2.3% | **49.5%** |
| 2.0 se — the floor | 23.0% | 15.9% | **38.8%** |
| 2.802 se — the MDE | 6.8% | 42.2% | **49.0%** |
| 4.0 se | 0.4% | 84.1% | **84.5%** |

**a. It is not monotone.** A candidate whose true margin is 2 se is promoted 38.8%
of the time; one at 1 se is promoted 49.5% of the time. **A better candidate is less
likely to be promoted than a worse one.** No coherent decision rule does that, and
it is not a tuning problem.

**b. It has never had the power its MDEs claim.** Every published MDE is
`2.802 x se`, where `2.802 = z(0.975) + z(0.80)` — an assertion of 80% power. The
decision rule delivers **49.0%** at that margin, of which 6.8 points come from the
hole, so legitimate power at the stated MDE is **42%**.

**c. Its size is 41% analytically, 28.3% measured** (`p2-gate-size-2026-09-20.md`);
the gap is the Brier, ECE and recall tolerances absorbing part of the fall-through.

## 2. What is already known, and is therefore not prospective

Stated plainly because rule 6 is about not re-specifying after the fact. **The whole
of §1 is analytic and was computed before this file was written.** It is arithmetic
over the code path, not a measurement of the gate. What is prospective is §5.

## 3. The change

`unresolved_against`'s band becomes `(0, floor]` — lower edge from `floor/1.5` to
**0**, upper edge from `floor*1.5` to **`floor`**:

```
UNRESOLVED  <=>  0 < |margin| <= floor
```

**The lower edge is the defect.** There was never an argument for a hole beneath the
band; a margin too small to resolve against its own floor is the definition of
unresolved.

**The upper edge follows from the MDE formula**, not from taste. `floor = 2 x se` is
approximately the 1.96 critical value of a two-sided 5% test, so "promote past the
floor" is that test, and its power at `2.802 se` is 78.9% — the 80% the MDE asserts.
Keeping the upper edge at `floor*1.5 = 3 se` would require a 3-sigma margin and give
**42%** power at the published MDE, which would make every MDE in the record wrong.

It also makes the code say what the documents already say — *"a margin smaller than
its noise floor is not a result"* (CLAUDE.md 7, CONTRIBUTING.md). The code currently
implements a band with a hole under it.

**`UNRESOLVED_BAND = 1.5` is retired.** Its docstring justifies it as floor
uncertainty — *"a floor estimated from three draws, whose own sampling spread is
~40% of its value"*. That is a real concern and this is the wrong treatment of it:
**floor uncertainty belongs in the floor, not in a dead band bolted around it.**
Widening the floor for its own df is #123's territory, and closing this hole is what
makes that work useful rather than harmful.

## 4. Scope — this touches recall as well as AUC

`unresolved_against` is called twice: on the AUC margin (`promotion.py:886`) and on
the recall margin (`promotion.py:859-868`). On recall it is reached with **either
sign**, and the same hole exists there: a regression smaller than `tolerance/1.5`
currently falls through both the REJECT test and the UNRESOLVED test and raises no
objection at all.

Under the new band a recall regression rejects once it exceeds its tolerance rather
than 1.5x its tolerance. **That is a behaviour change in the rejecting direction and
is reported as one**, not folded in silently.

## 5. How the result will be read — the prospective part

Three studies. None trains a model.

**5a. Size, re-measured.** `power_analysis.py --size` under the new rule, same
10,000 draws and the same four allocations. Falsification condition 1 applies.
**Predicted: ~1.6%**, from 2.3% analytic scaled by the 0.69 absorption the current
rule showed (28.3% measured against 41.0% analytic). Recorded so the measurement can
contradict it.

**5b. Power, which the size study did not cover.** The same simulation with the
candidate's mean displaced by `d x se` for `d` in 0 to 4, giving the operating
characteristic rather than a single point. Reported for the current and the new rule
together. Falsification condition 2 applies. **Predicted: ~62%** at one MDE — 78.9%
analytic times the same 0.79 absorption.

**5c. Every recorded verdict, re-checked**, with each change classified by whether
its margin exceeded its own floor. Falsification condition 3 applies.

## 6. What is not claimed

- **Not that the gate is then correct.** It fixes the decision rule's geometry. The
  floor it reads is still built on a candidate term measured on the wrong population
  (#123), and a correctly shaped rule around a wrong floor is still wrong.
- **Not that UNRESOLVED-under-the-null is acceptable.** It will rise to roughly half
  of null draws. Under a true null that is the correct verdict, but a weekly loop
  that returns UNRESOLVED half the time needs an operational answer and this does
  not provide one.
- **Nothing about `strict`**, which takes false-PROMOTE to 1.1% by a different
  mechanism and is not touched here.

---

## Result · 2026-09-20 · falsified on condition 2

Appended under rule 5; nothing above is edited.

[`p2-unresolved-band-result-2026-09-20.md`](p2-unresolved-band-result-2026-09-20.md).
The change in §3 was implemented, measured and **reverted**; `promotion.py` is
untouched.

Condition 2 asked that measured power at one MDE not stay below 70%. Measured
**1.6%**. At a true margin of 5 se it is 2.1%. The band is monotone and useless.

The cause is §4's scope, which was right that recall is touched and wrong about how
much: **98–100% of UNRESOLVED verdicts under the new band are raised by the recall
guard**, not by the AUC guard the change was about. `unresolved_against` serves two
different questions — *"is this improvement real?"* for AUC, *"is this regression a
veto?"* for recall — and the hole beneath the band is wrong for the first and right
for the second.

Condition 1 did not fire: size fell to 0.1%, better than the 1.6% predicted. **A
size-only study would have passed this change.**

§2's analytic non-monotonicity **was confirmed on the whole gate**: 28.9% at the
null, 34.8% at 1 se, 28.5% at 2 se.

**Reported as falsified, not re-specified.** Splitting the two questions is a
different change and needs its own terms.
