# P2 — the AUC guard promotes only past its floor · pre-registration · 2026-09-24

Written **before any change to `pipeline/src`**, per rule 6. Nothing here is adjusted
after the result. A result landing outside these terms is reported as falsified.
**No model is trained and nothing is promoted.**

Issue [#128](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/128).
Ollie chose the rule on 2026-09-24 from the four options costed in §2.

This is the third change on this code path, and the narrowest:

- [P2.2d](p2-unresolved-band-2026-09-20.md) applied `(0, floor]` to AUC **and**
  recall. It was falsified at 1.6% power, because 98–100% of its UNRESOLVED
  verdicts came from recall.
- [#131](p2-recall-one-sided-result-2026-09-24.md) made the recall guard
  one-sided. It landed today.

This file changes the **AUC call only**.

## How this will be read, fixed before the work starts

It **succeeds** if the implemented rule behaves exactly as the rule costed in §2,
and if no verdict moves to PROMOTE on an AUC margin that did not clear its floor.

It is **falsified** by any one of:

1. **The implementation's operating characteristic differs from the costed rule's
   in any cell.** After the change, every `shipped` row of `--options` must equal
   the pre-change `(0, floor]` row for the same allocation, floor and `strict`
   setting, at seed 7 and at seed 42. Common random numbers make that exact. A
   difference means the change touched something beyond the AUC call: recall,
   Brier, ECE or the verdict ordering.
2. **Any verdict moves to PROMOTE on an AUC margin that did not exceed its own
   floor.** This covers the recorded logs and every run in §2b.
3. **Any verdict changes whose AUC margin is ≤ 0.** Those reject at
   `promotion.py:808`, before the rule is reached.

**Size and power are stated in §5a and are not free parameters.** §2a already
measured them for this rule by substitution, and condition 1 requires the
implementation to reproduce them. They are reported together, per the lesson
P2.2e taught, and **power is read against the gate's 81% ceiling (#134), not
against the 80% the MDE asserts**. No AUC rule can reach 80% on the full gate, so a
criterion written against 80% would be falsified before it ran.

## 1. The defect

`unresolved_against` marks UNRESOLVED on `[floor/1.5, floor*1.5]`. So a positive AUC
margin below `floor/1.5` is outside the band, has not been rejected, and reaches
PROMOTE. The derivation is in P2.2d §1 and #128.

Measured on the whole gate after #131, 5 v 5, floor on the TESS slice, seed 7,
6,000 draws: PROMOTE is **33.7%** at the null, 39.7% at 1 se, **31.6% at 2 se** and
40.5% at the MDE. It is not monotone. A candidate twice as good as another is
promoted less often.

## 2. What is already known, and is therefore not prospective

**Everything in this section was measured before this file was written.** The rule
was substituted inside the study, and the gate was not changed. It is stated here so
that nothing below can later be presented as a prediction that came true.

**2a. The four options on the whole gate.**
`power_analysis.py --options --seed 7 --size-draws 6000`. Each option replaces only
the AUC call's rule, and recall keeps its own. "As recorded" is the seed sd the
trainer writes today, all missions pooled. "TESS #123" is the value #123 would
correct it to.

| allocation, floor | AUC rule | PROMOTE at the null | at 1 se | at 2 se | at the MDE | at 10 se | monotone |
|---|---|---:|---:|---:|---:|---:|---|
| 5 v 5, as recorded | shipped | 26.7% | 37.7% | 54.7% | 69.8% | 81.2% | yes |
| | `(0, floor/1.5]` | 16.9% | 47.2% | 71.4% | 79.2% | 81.2% | yes |
| | **`(0, floor]`** | **9.0%** | 34.2% | 63.5% | **76.6%** | 81.2% | yes |
| | `(0, floor*1.5]` | 2.8% | 16.7% | 47.0% | 67.9% | 81.2% | yes |
| 5 v 5, TESS #123 | shipped | 33.7% | 39.7% | 31.6% | 40.5% | 81.2% | **no** |
| | `(0, floor/1.5]` | 7.2% | 30.2% | 60.3% | 75.3% | 81.2% | yes |
| | **`(0, floor]`** | **1.8%** | 12.8% | 40.9% | **63.7%** | 81.2% | yes |
| | `(0, floor*1.5]` | 0.1% | 1.8% | 12.8% | 34.7% | 81.2% | yes |
| 5 v 1, as recorded | shipped | 30.8% | 45.3% | 31.9% | 16.0% | 63.6% | **no** |
| | **`(0, floor]`** | **0.0%** | 1.2% | 8.4% | **24.9%** | 63.6% | yes |

Seed 42 agrees within 2 pp in every cell, and the monotone column is identical.

What this settles:

- **Nominal size.** `(0, floor]` holds the nominal 2.3% of a one-sided test at
  `2 x se` only once #123 is fixed: 1.8%. Until then the live floor is too narrow,
  and size is 9.0%.
- **The order is measured, not argued.** Under `(0, floor]`, correcting the seed sd
  (#123) moves size from 9.0% to 1.8%, toward nominal. Under the shipped band the
  same correction moves it from 26.7% to 33.7%, which is what falsified #123's
  pre-registration. #123 is justified once this lands.
- **The ceiling.** PROMOTE at 10 se is 81.2% at 5 v 5 and 63.6% at 5 v 1 under every
  rule. The Brier and ECE tolerances are constants, and they reject 11.3% and 32.3%
  of candidates on noise alone. That is filed as
  [#134](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/134). On AUC
  alone, `(0, floor]` delivers 78.9% at the MDE, and 0.789 × 0.812 ≈ 64% is what the
  table shows.
- **`strict`**, at the null only, holds size at or under 1.5% under every option.
  The weekly refresh already passes `--strict` (`refresh_pipeline.py:470`). The
  manual CLI defaults off, and it is the only path that can `--promote`. Power under
  `strict` is not simulated: the displacement moves the TESS AUC, not the fold AUCs
  its paired-folds alarm reads.

**2b. On real runs.** `power_analysis.py --recheck --against
models/cv/champion-rebaselined-today/cv_summary.json --against
models/reference/champion-m5-k2-2026-09-19/cv_summary.json`, on `main` at `d723f3c`.
The "under `(0, floor]`" column comes from substituting the rule. **It is the
prediction for §5b, not a measurement of the implementation.**

| champion | run | AUC margin | floor | × floor | now | under `(0, floor]` |
|---|---|---:|---:|---:|---|---|
| rebaselined-today | `branches-20260809-drop-periodogram` | +0.01150 | 0.01923 | 0.598 | PROMOTE | UNRESOLVED |
| rebaselined-today | `stage8-control` | +0.01035 | 0.01901 | 0.544 | PROMOTE | UNRESOLVED |
| rebaselined-today | `stage8-propensity` | +0.00377 | 0.01468 | 0.257 | PROMOTE | UNRESOLVED |
| rebaselined-today | `stage8-synthetic` | +0.01573 | 0.01559 | 1.009 | UNRESOLVED | **PROMOTE** |
| M=5 re-baseline | `stage8-synthetic` | +0.00464 | 0.00966 | 0.480 | PROMOTE | UNRESOLVED |

**Four real runs promote through the hole today**, at 0.26–0.60 times their own
floors. `stage8-propensity` promotes on a margin a quarter of its floor.

**One run moves the other way, and it is disclosed here because it is the cost of
landing this before #123.** `stage8-synthetic` against the stand-in clears its floor
by 1.009×, 0.015734 against 0.015590. That floor is built from the candidate's
all-mission seed sd (#123) and a borrowed champion prior over one member. So this is
the rule working as specified, and condition 2 does not apply to it. But the
promotion is not one to act on:

- Under `--strict` it is UNRESOLVED, on three unacknowledged alarms: Kepler AUC fell
  0.042, the TESS slice holds 373 more rows than the champion's, and the run was
  trained from a dirty tree.
- Nothing in this change promotes anything. No `--promote` is run.

**2c. A missing floor is a pass, and this change keeps it so.**
`fc4f3515-resummarised` promotes on +0.00686 with **no AUC floor**: its variance
block carries no `seed_sd`. Both the shipped rule and `(0, floor]` return False on a
missing floor, so it was not costed and is not changed here. See §6.

## 3. The change

At the AUC call site only. `promotion.py:808` has already rejected any margin ≤ 0:

```python
margin = cand_auc - champ_auc
if floor.auc is not None and margin <= floor.auc:
    unresolved.append(f"gate AUC margin {margin:+.4f} does not clear its "
                      f"{floor.auc:.4f} floor — not a result (rule 7)")
```

So **UNRESOLVED ⇔ 0 < margin ≤ floor, and PROMOTE ⇔ margin > floor**, other criteria
permitting. Here is why each edge sits where it does:

- **Lower edge, `floor/1.5` → 0.** This is the defect. A margin too small to resolve
  against its own floor is the definition of unresolved.
- **Upper edge, `floor*1.5` → `floor`.** `floor = 2 × se`, so "promote past the
  floor" is a one-sided test at about 2.3%, with 78.9% power at the 2.802-se MDE on
  AUC alone. That is what the MDE formula assumes. Keeping `floor*1.5` needs 3 se
  and gives 34.7% at the MDE (§2a).
- **Stage 6's caveat 2** was that a floor from three draws has about 40% sampling
  spread. That concern is real, but it belongs **in the floor**, as a
  df-aware width, which is #123's territory. It does not belong in a dead band
  around the floor.

It also makes the code do what CLAUDE.md rule 7 and CONTRIBUTING.md already say: *a
margin smaller than its noise floor is not a result.*

`unresolved_against` and `UNRESOLVED_BAND` stay, and **serve recall only**; the
constant's comment says so. From here, the `unresolved_band` key in a promotion
log's `thresholds` describes recall alone.

## 4. Scope

AUC only. The following are not touched:

- recall's one-sided band (#131);
- the Brier and ECE constants (#134);
- the seed sd population (#123);
- `strict`'s default;
- the missing-floor path (§2c).

The live size after this lands is §2a's "as recorded" row, **9.0% at 5 v 5**. #123
is what takes it to 1.8%.

## 5. How the result will be read — the prospective part

**5a. Fidelity.** `--options --seed 7 --size-draws 6000` and `--seed 42`, after the
change. Each `shipped` row must equal the pre-change `(0, floor]` row. Condition 1.
Expected at seed 7, PROMOTE at 0 / 1 / 2 / 2.802 / 5 / 10 se:

| allocation, floor | expected |
|---|---|
| 5 v 5, as recorded | 9.0 / 34.2 / 63.5 / 76.6 / 81.2 / 81.2 |
| 5 v 5, TESS #123 | 1.8 / 12.8 / 40.9 / 63.7 / 81.1 / 81.2 |
| 5 v 1, as recorded | 0.0 / 1.2 / 8.4 / 24.9 / 62.0 / 63.6 |

With `strict`, at the null: 0.3%, 0.1% and 0.0% respectively. At seed 42 the
expected rows are those printed by the pre-change run at seed 42. The four option
rows collapse to one after the change, because the AUC call no longer reads
`unresolved_against`.

**5b. Every recorded verdict, and every run in §2b.** `--recheck --against` both
champions, after the change. Conditions 2 and 3 apply. Predicted:

- **The three recorded verdicts do not change.** The two Phase 1 arms gain an AUC
  objection beside their recall one, at 0.44× and 0.27× their floors.
- **Across both scans, exactly the five changes in §2b's table, and no others.**

**5c.** `make test`, with tests pinning three things: the hole closed, a margin
between `floor` and `1.5 × floor` promoting, and a margin exactly at the floor
reading UNRESOLVED.

## 6. What is not claimed

- **That the gate is then correctly sized.** It is 9.0% at 5 v 5 until #123 lands.
- **80% power at the MDE.** 63.7% is expected, against an 81.2% ceiling set by #134.
- **Anything about `strict`'s power or its default.**
- **That a missing floor is handled.** A candidate whose summary lacks `seed_sd` has
  no AUC floor and promotes on any positive margin (§2c). That is this issue's
  defect class, a margin with no floor read as a pass. It is left for its own issue
  rather than folded in.
- **That 5 v 1 is usable.** Whatever the rule, the gate against a single-member
  champion caps at 63.6%, and `(0, floor]` delivers 24.9% at the MDE. The M=5
  re-baseline should be the champion summary the gate reads.
