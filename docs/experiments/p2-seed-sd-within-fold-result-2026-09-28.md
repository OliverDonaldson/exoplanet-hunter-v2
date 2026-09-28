# P2 — the gate's seed sd, on the population it decides on · result · 2026-09-28

Read against the terms fixed in
[`p2-seed-sd-within-fold-2026-09-27.md`](p2-seed-sd-within-fold-2026-09-27.md),
committed as `5300575` on 2026-09-27 before any change to `pipeline/src` and merged as
[#138](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/138) (`43ecfbd`).
Issue [#123](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/123).
Nothing trained, nothing promoted, nothing under `models/` written.

**Not falsified: none of the four conditions fires.** The AUC floor now reads
`gate_roc_auc_seed_sd`, the within-fold TESS term, on both sides. Every simulation
decides byte for byte as it did before the change. On disk, exactly the changes §2c
listed occur and no recorded verdict moves. A multi-member summary without the term is
refused, and a single-member one is not.

Reproduction, each at `--seed 7 --size-draws 6000` where it takes them:
`python pipeline/scripts/power_analysis.py --recheck --corrected --against
models/cv/champion-rebaselined-today/cv_summary.json --against
models/reference/champion-m5-k2-2026-09-19/cv_summary.json`, then `--options`, `--size`,
`--operating-curve` and `--estimators`. Plain `--recheck` now raises (§6).

## 1. What was run

- **Before** is `pipeline/src` at `43ecfbd`, whose tree is identical to `5300575`. The
  five commands of §5b ran on it before `pipeline/src` was touched, stdout only, with the
  imported `promotion.py` checked.
- **After** is the change below. The same five commands ran on it and were compared with
  `cmp`.
- For condition 3, every `cv_summary.json` on disk was re-decided against both champions
  under both codes, outside the tracked output, as the #128 result did: 64 pairs. Before
  ran with the archived `pipeline/src` first on `PYTHONPATH`, with `gate.__file__`
  checked. After ran with the term injected the way `--corrected` injects it.

## 2. The change

- **`eval/comparison.py::gate_auc_seed_sd`** is §3's estimator: each member's ROC-AUC
  on a fold's TESS rows, the variance across members at ddof 1, the mean over folds on
  F(M−1) df, then the square root over c4(df+1), with c4 from `math.lgamma`. It returns
  `gate_roc_auc_seed_sd` and `gate_roc_auc_seed_df`, and None with 0 df below two
  members. It raises on a non-finite member score, on a missing `fold` column with two
  or more members, and on a single-class fold slice.
- **It is merged beside `pooled_member_draws`** at all three sites: `train.py`,
  `train_branches.py` and `summarise_scored`.
- **`decision_floor` reads it on both sides.** A summary recording two or more members
  without it, or with it None, raises before any verdict. One member, or no variance
  block, is unchanged. A champion with no variance still borrows `POOLED_SEED_SD`.
- **`variance.seed_sd` keeps its value and its writers.** The trainers' comments now
  say it is all missions, within fold.

Three things go beyond the letter of §4, and are stated so they are not read as silent:

1. **The refusal names the run through a new keyword, `names`**, on `evaluate_promotion`
   and `decision_floor`. A summary records no path of its own, so the two callers that
   know one pass it: `promotion_gate.py` and `power_analysis.py --recheck`. Every other
   caller's message says "the candidate summary" or "the champion summary".
2. **A TESS row with no fold also raises.** `groupby` drops a null key, which would
   shrink the df without saying so. No run on disk has one. A frame with no TESS rows
   returns the null that `pooled_member_draws` returns, rather than raising.
3. **Condition 1 is checked inside `tess_seed_sd` and reported on stderr**, not as a
   column in the `--corrected` table. A column would change the stdout condition 2
   requires to be byte-identical. `tess_seed_sd` now computes the library term on the
   frame it reads, and raises past 1e-12 relative. So every `--recheck --corrected` run
   re-tests condition 1 on every run it measures.

## 3. Condition 1, the term: does not fire

On all 21 measurable runs the library term equals `tess_seed_sd(...).corrected`. The
largest relative difference is **1.2e-15**, against the 1e-12 limit. That is the gap
between `math.lgamma` and scipy's `gammaln`, plus summation order. The stderr line:

> `#123 condition 1: the library term matches on 21 runs, largest relative difference
> 1.2e-15 against a 1e-12 limit`

The re-baseline's term is **0.004059823** from both, which §2a printed as 0.00406. A
synthetic run pins the same equality where no run directories exist, in CI
(`test_the_study_term_is_the_library_term`).

## 4. Condition 2, the gate against the emulation: does not fire

All five outputs are byte-identical before and after:

| command | bytes | sha256, first 16 |
|---|---:|---|
| `--recheck --corrected`, both champions | 6,897 | `c4bde2a1cd3c5dd3` |
| `--options` | 3,435 | `96f2f93f47386c12` |
| `--size` | 1,309 | `94a7e90c154d4472` |
| `--operating-curve` | 1,886 | `ac77840a1cd1339e` |
| `--estimators` | 951 | `f8308ab934663acf` |

The `--corrected` output reproduces §2a exactly: 23 rows, 21 measurable, the
architecture ratios 0.86 and 0.98. The `--estimators` output reproduces §2b cell for
cell.

## 5. Condition 3, verdicts on disk: does not fire

Across all 64 run × champion pairs, exactly six change. They are §2c's five rows, one of
which (`branches-20260808-capacity`) changes against both champions:

| champion | run | before | after |
|---|---|---|---|
| rebaselined-today | `stage8-synthetic` | PROMOTE, 1.009× its 0.01559 floor | UNRESOLVED, 0.797× its 0.01974 floor |
| rebaselined-today | `fc4f3515-resummarised` | PROMOTE, no AUC floor | UNRESOLVED, 0.460× its 0.01494 floor |
| rebaselined-today | `branches-20260807-shared` | REJECT | refused, no member scores |
| rebaselined-today | `branches-20260808-capacity` | REJECT | refused, no member scores |
| M=5 re-baseline | `branches-20260807-shared` | REJECT | refused, no member scores |
| M=5 re-baseline | `branches-20260808-capacity` | REJECT | refused, no member scores |

The other 58 keep their verdicts, although every multi-member floor moves.
`stage105-control`'s floor narrows 0.02048 → 0.01597, taking its margin from 0.732× to
0.938× of it, and it stays UNRESOLVED, as §2c said. **The three recorded verdicts do not
change**: UNRESOLVED, UNRESOLVED and REJECT.

## 6. Condition 4, the refusal: does not fire

Plain `--recheck` now raises at the first recorded log, before printing a verdict:

> `ValueError: models/phase1/arm-c-control/cv_summary.json records 3 members per fold
> and no gate_roc_auc_seed_sd, the TESS-slice seed sd the AUC floor has read since
> #123. Re-summarise its predictions to a NEW path, never over the original: python
> pipeline/scripts/evaluate.py summarise --predictions <run>/predictions.parquet
> --protocol oof --out <new dir>/cv_summary.json`

On the summaries exactly as they sit on disk, the gate now refuses the 55 of 64 pairs in
which the candidate or the champion records two or more members. It decides the 9 in
which both record fewer. No multi-member summary is decided, and no single-member one is
refused. `promotion_gate.py` was not run on a real run directory.

## 7. What re-summarising the re-baseline costs, measured

After this change the re-baseline must be re-summarised before #114 can adopt it. That
was done to a scratch path, not under `models/`:

```
python pipeline/scripts/evaluate.py summarise \
  --predictions models/reference/champion-m5-k2-2026-09-19/predictions.parquet \
  --protocol oof --exclude-unresolved --out <new dir>/cv_summary.json
```

- **`--exclude-unresolved` is required.** Five rows have no mission in
  `viewset_scalars.parquet`, and four are TESS in the label table. So the re-summarised
  TESS slice holds **2,367 rows, not the trainer's 2,371**. That is the same count as the
  stand-in and both Phase 1 arms.
- **So its term is 0.004074**, not 0.004060, and its TESS AUC is 0.920994, not
  0.921102.
- **`summarise_scored` writes no `folds` block.** So against it `paired_folds` and
  `blocked_contrast` return None, and the paired-folds alarm that `--strict` reads
  cannot fire.

That is a measurement, not a decision. Whether #114 adopts the file as it stands is a
separate choice.

## 8. The gate as it now stands

- **At 5 v 5 on this estimator**, size is **2.1%** (1.8% on the exact term) against a
  nominal 2.3%. Power at the MDE is **63.2%**, against the 81.2% ceiling of #134.
- **At the refresh's 3 v 3**, size is 1.8% and power 55.1%. Both allocations are §2b's,
  reproduced byte for byte.
- **Against the single-member stand-in**, the champion borrows `POOLED_SEED_SD` over one
  member as before. The candidate's term barely moves that floor: 0.01256 → 0.01292 on
  the re-baseline's term. Size is 0.0%, power at the MDE 24.9%, and the ceiling 63.6%.
  #114 is what moves it.
- The fast suite passes; mypy reports the same 78 errors, and the comment share is
  24.90%.

## 9. What is not claimed

- **Anything about single-member candidates** (#137), which still have no AUC floor, or
  about the Brier and ECE constants (#134).
- **That recall's term is right.** It is fold-pooled on M−1 df and not c4-corrected, so
  at M=3 `E[s] = 0.886σ`.
- **That the console's floor is right.** `api/app/routes/model.py:95` still displays
  `seed_sd`. That is inert while the served champion is single-member.
- **That the re-baseline is ready to be champion.** §7 is what that costs.
