# P2 — the AUC guard promotes only past its floor · result · 2026-09-24

Read against the terms fixed in
[`p2-auc-floor-rule-2026-09-24.md`](p2-auc-floor-rule-2026-09-24.md). Those were
pushed as `a5bbee3` and opened as
[#135](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/135) at
09:58 UTC, before `pipeline/src` changed. Nothing trained, nothing promoted.

**Not falsified: none of the three conditions fires, and every prediction in §5 of
the pre-registration held exactly.**

Reproduction:
`python pipeline/scripts/power_analysis.py --options --seed 7 --size-draws 6000`,
the same at `--seed 42`, and
`--recheck --against models/cv/champion-rebaselined-today/cv_summary.json
--against models/reference/champion-m5-k2-2026-09-19/cv_summary.json`.

## 1. What was run

- **Before** is the pre-registration's own output from `a5bbee3`. Its `pipeline/src`
  is identical to `main` at `d723f3c`.
- **After** is the change: one call site in `promotion.py`, and a comment on
  `UNRESOLVED_BAND` saying it now serves recall only. The imported module was
  checked.
- The same three commands were run before and after.

## 2. Condition 1, fidelity: does not fire

After the change, every row `--options` prints is the implemented rule, because the
AUC call no longer reads `unresolved_against` and the substitution has nothing to
replace. That is 6 configurations × 4 rows at each seed. **All 48 rows at seed 7 and
all 48 at seed 42 are identical, character for character**, to the pre-change
`(0, floor]` row for the same configuration. The change touched nothing but the AUC
call.

The gate as shipped now, seed 7, PROMOTE at 0 / 1 / 2 / 2.802 / 5 / 10 se:

| allocation, floor | non-strict | strict, at the null |
|---|---|---:|
| 5 v 5, as recorded | 9.0 / 34.2 / 63.5 / 76.6 / 81.2 / 81.2 | 0.3% |
| 5 v 5, TESS #123 | 1.8 / 12.8 / 40.9 / 63.7 / 81.1 / 81.2 | 0.1% |
| 5 v 1, as recorded | 0.0 / 1.2 / 8.4 / 24.9 / 62.0 / 63.6 | 0.0% |

**Monotone in every configuration.** Before the change it was not, at 5 v 5 on the
TESS-slice floor or at 5 v 1.

## 3. Conditions 2 and 3, verdicts on disk: do not fire

**The three recorded verdicts are unchanged**, as predicted. Both Phase 1 arms now
carry an AUC objection beside their recall one:

- `arm-c-control`: *gate AUC margin +0.0076 does not clear its 0.0173 floor*
- `arm-d-difference`: *gate AUC margin +0.0060 does not clear its 0.0220 floor*

**Across both scans, exactly the five changes the pre-registration predicted, and no
others:**

| champion | run | × floor | before | after |
|---|---|---:|---|---|
| rebaselined-today | `branches-20260809-drop-periodogram` | 0.598 | PROMOTE | UNRESOLVED |
| rebaselined-today | `stage8-control` | 0.544 | PROMOTE | UNRESOLVED |
| rebaselined-today | `stage8-propensity` | 0.257 | PROMOTE | UNRESOLVED |
| rebaselined-today | `stage8-synthetic` | 1.009 | UNRESOLVED | PROMOTE |
| M=5 re-baseline | `stage8-synthetic` | 0.480 | PROMOTE | UNRESOLVED |

The scan prints positive margins only. The runs with a margin ≤ 0 are not printed,
and they did not change either. All 64 run × champion pairs were re-decided under
both codes outside the tracked output. The result also follows from
`promotion.py:808`, which rejects those runs before the changed line.

**The one move to PROMOTE clears its floor**, 0.015734 against 0.015590, so
condition 2 does not apply. It is the cost the pre-registration disclosed in §2b,
of landing this before #123: that floor is built on an all-mission seed sd #123
says is understated. Under `--strict` the run is UNRESOLVED, on three
unacknowledged alarms. Nothing was promoted.

## 4. One test had been passing for the wrong reason

`test_noisier_candidate_no_longer_quietly_earns_a_wider_pass` asserted UNRESOLVED.
Its docstring credited the candidate's wider **recall** band. Re-run against `main`,
that is not what happened:

- its recall drop, −0.0370, was 0.55× its 0.0673 floor and raised nothing;
- the UNRESOLVED came from the **AUC** band's upper edge, with a margin of 0.0070 at
  1.43× its 0.0049 floor.

That upper edge is exactly what this change removes, by pre-registration. So the
test had been checking the wrong criterion since it was written in `839ff8c` on
2026-08-17.

It has been rewritten to test its stated claim: a recall drop at 0.89× its floor,
with the AUC margin clear of its own floor. The result is UNRESOLVED, raised by
recall and not by AUC. The new test,
`test_the_auc_guard_promotes_only_past_its_floor`, pins five points around a floor
of exactly 0.0625, and it fails against `main` at its first case.

## 5. The gate as it now stands

- **Size at 5 v 5 is 9.0% on the floor the trainer writes today**, down from 26.7%.
  **#123 is what takes it to 1.8%**, and it is now justified: under this rule,
  correcting the seed sd moves size toward nominal instead of away from it.
- **Power at the MDE is 76.6%, or 63.7% after #123**, against an 81.2% ceiling that
  no AUC rule can pass, because the Brier and ECE tolerances are constants
  ([#134](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/134)).
- Carried in [`known-limits.md`](../known-limits.md).

## 6. What is not claimed

- **That a missing floor is handled.** `fc4f3515-resummarised` still promotes on
  +0.00686 with no AUC floor, exactly as before.
- **That 5 v 1 is usable.** It caps at 63.6%, and delivers 24.9% at the MDE.
- **Anything about `strict`'s power.**
