> Moved verbatim from `docs/roadmap.md` §3.3 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.3 Stage 5 — the candidate view set, rebuilt (2026-08-08)

Done, cold, as budgeted: **5,346 rows at 2001/201, 309 MB, 95 minutes** for
7,174 catalogue rows. `ViewSetArrays.validate()` reports it well-formed, every
view matches training's `VIEW_SHAPES`, and **run 3's fold-0 checkpoint scores it**
— which is the whole point, since no post-run-1 model could touch the old set.
The 401 MB of orphaned cache was left in place; nothing reads it, and deleting
it is not this task.

Every one of the 7,174 rows is accounted for: 5,346 built, 1,803 no FITS, 17
preprocess errors, 8 with no ephemeris. Sources are 3,929 SPOC 2-minute, 719
FFI, 698 Kepler; DV usable on 64%, RUWE on 85%.

**The row count moved 5,347 → 5,346, and the three rows reconcile exactly
against the refresh.** TIC 60520371 and TIC 160476088 were dispositioned **FP**
on 2026-08-08 and so left `candidates.parquet` for `labels.parquet` — they are
precisely the two rows behind the catalogue's 5,703 → 5,705 growth. TIC
443534757 is a new **PC**. Net −2 +1.

Stated rather than assumed, because a candidate set that quietly changed size is
indistinguishable from one built over a population nobody chose — and per the
handover a rebuilt set **will** trip the gate's row-count alarm against run 3
and the capacity arm, which were measured on the previous catalogue. That alarm
is correct and should not be silenced.

**Stage 7's success criterion — recorded as stage 2(b)'s, re-specified
2026-08-05, and not renumbered inside.** It read
*"corr(prob, n_transits) must leave zero and the 26.4% control-arm host-pass
rate must fall"*, with a companion requirement that the baseline correlation
fall from +0.211. That criterion is now split, because its two halves are not
the same kind of measurement:

- **The control-arm host-pass rate is the criterion.** It is measured on real
  hosts with *no injection*, so a pass means the model scored the star rather
  than a transit. No label structure enters it and nothing about the catalogue
  can explain it away. **26.4% must fall.**
- **The baseline correlation is retired as a gate** and kept only as a reported
  diagnostic. Driving it to zero would move the model away from its own labels
  — see stage 3 [now 8].
- **The transit-count correlation is reported, not gated.** Its zero point is
  **−0.048** against transits captured, not the −0.003 that was measured against
  transits predicted; and the labels themselves sit at −0.073, so there is no
  defensible target value to demand.

The clean test of the unfolded branch is **injection-recovery on matched hosts
with observation baseline held constant**, which removes the label confound
entirely. Build that harness when 2(b) [now stage 7] is run.

**Stage 6 — recall variance + re-baseline.** *(next)*
`recall @1% FPR` is the criterion that has rejected all four arms of stage 4 —
run 3 on **0.145 vs 0.307** — and it has **no variance estimate at all**. AUC's
noise floor was measured (`seed_sd 0.0081`, `fold_sd 0.0094`) and "a margin under
~0.009 is not a decision" adopted from it; the statistic that does the actual
rejecting never got the same treatment.

`_variance_decomposition` reads only `model_roc_auc`, and the per-member metrics
recorded beside it come from `classification_metrics`, whose `.recall` is recall
at threshold 0.5 — **not** the gate's statistic. The gate's `recall_at_1pct_fpr`
lives in `SliceMetrics` (`eval/comparison.py`), verified exact against sklearn
and a brute-force threshold sweep. So each member records its own
`recall_at_1pct_fpr` alongside its AUC, and the summary reports `recall_seed_sd`
/ `recall_fold_sd` beside the existing pair. **Purely additive to
`cv_summary.json`** — the promotion gate reads named keys and is unaffected.

**Three estimates, not one, because a fold is the wrong population.** TESS holds
2,399 rows at a 0.552 base rate, so a fold's TESS test slice carries ~215
negatives and its 1% FPR cut is **two rows** — the statistic is set by where the
third-highest-scoring negative lands, and its spread says more about that than
about the model. The gate reads the *pooled* out-of-fold set: ~1,074 negatives,
a cut of ten. So `predictions.parquet` gained one uncalibrated score column per
ensemble member, and stacking member *i*'s column across folds re-forms a
complete out-of-fold prediction set for that member alone. Three members, three
independent draws of **the number the gate actually reads**, at the cost of three
float columns and no extra training or inference.

| reported | what it is |
|---|---|
| `recall_seed_sd` / `recall_fold_sd` | the whole fold, every mission — mirrors the AUC pair exactly |
| `gate_recall_seed_sd` / `gate_recall_fold_sd` | that fold's TESS rows alone; **coarse, and an upper bound** |
| `pooled_gate_recall_seed_sd` | spread of the three pooled-TESS draws — **the primary estimate** |

Then the re-baseline itself: HEAD, `--n-models-per-fold 3`, 5 folds, over the
**unchanged** training shards (5,426 rows: 2,500 Kepler, 2,399 TESS, 527 K2).
Same rows as run 3 and the capacity arm, so the only differences are the three
code changes and the gate's row-count alarm should stay silent. It is the
control for every subsequent stage — three training-path changes have landed
since run 3, so nothing measured before 2026-08-08 is a baseline — and it returns
the recall noise floor for free.

**This replaces the queued capacity repeat**, which was specified against the
capacity arm's architecture. That architecture no longer exists on HEAD, so
running it now would measure the rebuilt unfolded branch rather than the capacity
arm — the same trap as `init_filters=22` being re-derived against run 3. This
measures the thing underneath instead, on the architecture that exists, for one
run instead of three.
