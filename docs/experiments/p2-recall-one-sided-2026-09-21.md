# P2 — the recall guard objects to good news · pre-registration · 2026-09-21

Written **before any code changes**, per rule 6. Nothing here is adjusted after the
result; a result landing outside these terms is reported as falsified.

**No model is trained and nothing is promoted.** Issue
[#131](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/131), following
the attribution measured in
[`p2-unresolved-band-result-2026-09-20.md`](p2-unresolved-band-result-2026-09-20.md).

This is deliberately the **minimal** change. The two pre-registrations before it were
falsified for the same reason — each bundled a bug with a design question. This one
fixes the bug only.

## How this will be read, fixed before the work starts

It **succeeds** if the gate stops raising a recall objection to a recall
*improvement*, if that is visible as higher power at a large true margin, and if no
rejection changes.

It is **falsified** by any one of:

1. **Any verdict moves from REJECT to PROMOTE.** The change must not touch the
   rejection path at all. One that does is an implementation error, not a result.
2. **Measured PROMOTE at a 5-se true AUC margin does not rise by at least 5 pp**
   (from 71.6%). The blocking would then not be material and the change not worth
   making.
3. **Any verdict changes where the recall margin was negative.** The change is
   scoped to improvements; a regression must be handled exactly as before.

## 1. The defect

`unresolved_against` tests `abs(margin)`, and the recall guard calls it with a margin
of either sign (`promotion.py:860`). A recall margin between `tolerance/1.5` and
`tolerance*1.5` therefore marks UNRESOLVED **whichever way it went**.

Measured on the re-baseline's own summary, with a +0.02 AUC improvement — 4.2x its
floor — and nothing varying but recall:

| recall move | verdict |
|---|---|
| unchanged | PROMOTE |
| improved by 0.5x its floor | PROMOTE |
| **improved by 1.0x its floor** | **UNRESOLVED** |
| **improved by 1.4x its floor** | **UNRESOLVED** |
| improved by 2.0x its floor | PROMOTE |
| regressed by 1.4x its floor | UNRESOLVED — correct |

Non-monotone in recall, and there is no reading of the gate's purpose under which
"recall improved by a medium amount" is a stop-and-ask. The project's own rule is
that **recall cannot promote, it can only veto**; an improvement is not a veto.

## 2. What is already known, and is therefore not prospective

The table above and the attribution rates in §5 were measured before this file was
written. What is prospective is §4.

## 3. The change

One line, at the call site, leaving `unresolved_against` untouched so the AUC path is
not affected:

```python
if margin < 0 and unresolved_against(margin, recall_tolerance):
```

The REJECT branch above it already tests `margin < -recall_tolerance` and is not
touched. **`unresolved_against` itself is not changed**, so #128 remains open and
entirely separate.

## 4. How the result will be read — the prospective part

`power_analysis.py --operating-curve`, 6,000 draws at 5 v 5, before and after,
with each UNRESOLVED attributed to the criterion that raised it and — for recall —
to the direction the recall margin moved.

Predicted, from the shares measured in §5:

| true AUC margin | PROMOTE now | predicted after |
|---|---:|---:|
| 0 — the null | 30.4% | **33–35%** |
| 2.802 se — the MDE | 36.4% | **42–45%** |
| 5.0 se | 71.6% | **78–80%** |

Plus every recorded verdict re-checked, classified by the direction of its recall
margin. Falsification conditions 1 and 3 apply there.

## 5. Size will get worse, and that is the correct outcome

At the null, 28% of UNRESOLVED verdicts are raised by a recall *improvement*; at a
5-se margin, 46%. Removing that objection converts most of those draws to PROMOTE,
so **the gate's measured size under a true null will rise, from 30.4% to a predicted
33–35%.**

**This is pre-registered as expected and is not a falsification.** The blocking being
removed was never a guard — it was a bug acting as a brake, stopping roughly 8% of
candidates at random with respect to their actual merit. A gate whose size depends on
a bug is not correctly sized; it is wrong twice in opposite directions.

The gate's size is the AUC band's job, it is broken (#128), and it is not fixed here.
Leaving a defect in the recall guard because it accidentally suppresses false
promotions would be exactly this project's named defect class — a number that looks
right for the wrong reason.

## 6. What is not claimed

- Nothing about the AUC band (#128), the seed sd population (#123), or `strict`.
- Nothing about whether the recall guard's *threshold* is at the right place; only
  that it should not fire on an improvement.
- The gate remains badly sized after this lands, and by a slightly larger margin
  than before.
