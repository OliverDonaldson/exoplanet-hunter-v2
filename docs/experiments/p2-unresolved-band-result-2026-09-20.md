# P2 — the UNRESOLVED band, measured · result · 2026-09-20

**The change pre-registered in
[`p2-unresolved-band-2026-09-20.md`](p2-unresolved-band-2026-09-20.md) is
falsified on condition 2.** It was implemented, measured, and reverted. Nothing
in `pipeline/src` changed.

Two findings stand, and the second is larger than the one this study set out to
test.

Reproduction: `python pipeline/scripts/power_analysis.py --operating-curve`.

## 1. What was run

The same null simulation as [`p2-gate-size-2026-09-20.md`](p2-gate-size-2026-09-20.md),
extended with a displacement: the candidate's gate-slice ROC-AUC is bumped by
`d x se`, giving a **true** margin of `d se` rather than only the null. Both bands
are measured, the shipped one and the proposed one, by substituting
`unresolved_against` inside the study rather than by changing the gate.

Each UNRESOLVED is attributed to the criterion that raised it. That attribution is
what the rates alone hide, and it is where the finding turned out to be.

5 v 5, seed sd 0.00372, margin se 0.00235, floor 0.00471, MDE 0.00659, 4,000 draws.

## 2. The shipped band is not monotone — measured, not just derived

| true margin | PROMOTE | UNRESOLVED | REJECT | UNRESOLVED raised by |
|---|---:|---:|---:|---|
| 0 — the null | **28.9%** | 15.0% | 56.1% | recall 55% / auc 45% |
| 1.0 se | **34.8%** | 38.8% | 26.5% | recall 34% / auc 66% |
| 1.33 se — band edge | 32.6% | 46.8% | 20.6% | recall 31% / auc 69% |
| 2.0 se — the floor | **28.5%** | 57.0% | 14.5% | recall 28% / auc 72% |
| 2.802 se — the MDE | 35.7% | 51.8% | 12.4% | recall 31% / auc 69% |
| 5.0 se | 69.8% | 17.9% | 12.2% | recall 89% / auc 11% |

**A candidate whose true margin is 2 se is promoted 28.5% of the time. One at 1 se
is promoted 34.8% of the time. One with no improvement at all, 28.9%.**

The pre-registration derived this analytically; it is now measured on the whole
gate, and it is worse than the derivation suggested, because a genuine 2-se
improvement is no more likely to be promoted than nothing at all.

**Power at the MDE is 35.7%** against the 80% every published MDE asserts.

## 3. The proposed band is falsified — condition 2

| true margin | PROMOTE | UNRESOLVED | REJECT | UNRESOLVED raised by |
|---|---:|---:|---:|---|
| 0 — the null | 0.1% | 42.9% | 57.0% | recall 98% / auc 2% |
| 2.0 se — the floor | 0.9% | 82.6% | 16.4% | recall 99% / auc 1% |
| 2.802 se — the MDE | **1.6%** | 84.0% | 14.4% | recall 99% / auc 1% |
| 5.0 se | **2.1%** | 83.8% | 14.2% | recall 100% / auc 0% |

Condition 2: *"Measured power at a true margin of one MDE stays below 70%."*
Measured **1.6%**. A candidate more than twice the MDE promotes 2.1% of the time.
The band is monotone, and useless.

**Reported as falsified, not re-specified, per rule 6.** The change was reverted;
`promotion.py` is untouched. Only the study that measured it is kept.

**Condition 1 did not fire, and that is the instructive part.** It asked whether
size stayed above 5%; size fell to **0.1%**, better than the 1.6% predicted. A
size-only study would have passed this change and shipped a gate that cannot
promote anything. Chagas et al. §4.2 report size and power together for this
reason; pre-registering both is what caught it.

## 4. The finding that matters: the recall guard is the binding constraint

Read the attribution column. Under the proposed band, **98–100% of UNRESOLVED
verdicts are raised by the recall guard**, not by the AUC guard the change was
about. Under the *shipped* band at a 5-se margin it is **89%**.

The cause is that `unresolved_against` serves two questions that are not the same
question:

- **AUC** asks *"is this improvement real?"* — a margin must exceed its floor to
  promote, and one below it is genuinely unresolved.
- **Recall** asks *"is there a regression large enough to veto?"* — a small move is
  **not** a veto, and treating it as unresolved blocks everything.

The hole under the band is wrong for AUC and right for recall. A single rule cannot
be right for both, and the shipped band is a compromise that is wrong for each in a
different direction.

With recall's seed sd at 0.03028, the recall margin's se at 5 v 5 is 0.01915 and its
floor 0.0383 — exactly 2 se. So under any rule that calls `|margin| <= floor`
unresolved, **95% of all candidates are unresolved on recall alone, whatever their
AUC does.**

## 5. What this does not establish, and what is not done

- **No fix is proposed here.** Splitting the two questions is a different change
  from the one pre-registered, and re-specifying this file to cover it is exactly
  what rule 6 forbids. It needs its own terms.
- The AUC hole remains open and #128 remains open. Closing it is still necessary
  and is still not sufficient.
- Nothing was measured about `strict`, about allocations other than 5 v 5, or about
  the recorded verdicts — study 5c was not reached, because the change it would have
  re-checked is not being made.
