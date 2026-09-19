# P2 — champion re-baseline at M=5, with K2 · pre-registration · 2026-09-19

Written **before the run starts**, per rule 6. Nothing here is adjusted after the
result; a result landing outside these terms is reported as falsified.

This is an instrument calibration, not a challenger. **No performance claim is
made and nothing is promoted.** `models/registry.json` is not touched, and the
served champion stays `ca906040`.

## 1. Why

`decision_floor` computes the standard error of a *difference*:

```
se(delta) = sqrt( sd_c^2 / n_c + sd_i^2 / n_i )
```

`seed_sd` is written only by the trainers (`training/train.py:658`).
`summarise_scored` builds its variance block from `pooled_member_draws`, which
returns only `pooled_gate_recall*` keys (`eval/scoring.py:309-310`) — measured:
`models/cv/fc4f3515…` carries `seed_sd`, and `models/cv/fc4f3515-resummarised`,
the same run re-summarised, does not. So no amount of re-scoring gives the
champion a variance block, and `champion-rebaselined-today` carries
`n_models_per_fold: 0`.

The gate therefore takes the borrowed branch at `promotion.py:364`,
`term += POOLED_SEED_SD**2 / 1`. The allocation it faces is **5 against 1**, not
the 5-against-5 the published MDEs assume.

## 2. What will be run

```
python -m exoplanet_hunter.training.train data=full train.n_models_per_fold=5
```

on the shard set at `data/processed/tfrecords` — 5,380 rows, measured by joining
its index to `labels.parquet` on `tic_id`: **Kepler 2,480, TESS 2,371, K2 527**,
2 unmatched. No pinned `fold_assignment`, so every row is used.

K2 is therefore included by construction, consistent with the 2026-09-14
decision that the clean run includes K2. The shard set is not rebuilt and no DVC
command is run.

The run is written by `train.py` to `models/cv/<mlflow_run_id>/` and **moved out
of `models/cv/` on completion**, because the weekly gate selects the newest
`models/cv/*/cv_summary.json` excluding only `control-lane`, and
`decide_training` skips training below 25 new labels — in which case this run
would be selected as a candidate and gated against the champion it is.

## 3. How the result will be read

**Primary criterion.** The run's `cv_summary.json` carries
`summary.variance.seed_sd`, finite, at `n_models_per_fold: 5`. That is the whole
deliverable. There is no accuracy claim.

**The MDE that follows.** With `sd_boot = 0.0021` (P2.1 §1, the TESS ROC-AUC
sampling sd) and the measured `sd_seed` on both sides:

```
MDE = 2.802 * sqrt( sd_c^2/5 + sd_i^2/5 + sd_boot^2 )
```

Pre-registered expectation: **≈0.0125**, if the measured `seed_sd` lands near the
0.0062 pooled dual-view prior. The figure the gate faces today is **0.0199**.

**What falsifies this.** If the measured champion `seed_sd` is **above 0.0093** —
the single-arm dual-view figure P2.1 measured before pooling — then the 0.0062
pooled prior was optimistic as a champion-side stand-in, and every floor the gate
has read was too **narrow**, the opposite of #94's direction. That is reported as
a falsification of the prior, the MDE table in #114 is redone against the
measured value, and the re-baseline is *not* re-run to get a better number.

**Estimand, stated so it is not misread later.** This run is a different set of
weights from `ca906040`. The two-term `se(delta)` compares one training
*procedure* against another, both sides means over seeds. It does not compare a
candidate against the fixed number the deployed artefact scored — that is a
different question, with no champion variance term, and it is the superseded
one-term formula. Neither is wrong; they answer different things, and this run
supports the first.

**Alarms — neither is acknowledged in advance.** The K2 mismatch ("only the
candidate scored K2") and the TESS row-count drift (champion 2,367 against the
current 2,371) are both expected to stop firing, because this run includes 527
K2 rows and trains on the current TESS population. If either still fires, that
is a finding and is reported as one. Pre-acknowledging them would be
warn-and-continue one level up.

**Wall-clock is recorded.** No timing for a dual-view CV run exists anywhere in
this record. The nearest comparator is 4h44 for 5 folds x 3 members
(`stage-10-5-ensemble.md:236`), and W10 makes that a floor rather than an
estimate. Per-fold times are recorded whatever they are.

## 4. What this run does not do

It does not become servable. A run past one member per fold writes
`model_<i>_cnn_dualview.keras` and no bare checkpoint
(`training/mlflow_utils.py:224`), and `ScoringEnsemble` loads the bare name only
— so this artefact can be *gated* (the gate reads `cv_summary.json`) but cannot
be served or scored on the injection instrument until #113 lands. That is known
before the run rather than discovered after it.

## 5. Reverses a recorded decision

`promotion.py:413-417` records the opposite choice: *"Re-baselining the reference
to clear this is a real option and is not taken, because it would move the
comparison every past result was read against."* That cost is accepted
deliberately — every comparison made against the borrowed-prior floor is
superseded, and the 5-against-1 allocation is the reason. Dated row in
`docs/decisions.md`, and the comment is corrected in the same change.

---

## Correction · 2026-09-20 · the seed sd this run measured is on the wrong population

Appended under rule 5; nothing above is edited.

This pre-registration expected the MDE to fall to ≈0.0125 once the champion
carried its own variance block. It fell further, and **not for the reason
expected**. The `seed_sd` this run wrote — 0.00225 — is the mean within-fold
spread of per-member **all-mission** ROC-AUC (`train.py:551`), while the gate
decides on the fold-pooled **TESS** slice. Measured on this run's own
`predictions.parquet`, the TESS figure is **0.00372**, 1.65x larger.

So §1's argument stands — the gate faces 5-against-1 and the published MDEs
assume 5-against-5 — but the number this run supplies for the candidate side
cannot be used as it is. Wiring this artefact in as champion unchanged would
give an AUC floor of **0.00285** where the corrected term supports **0.00471**:
too narrow, for a different wrong reason than the one this run was written to
fix.

The falsification condition — *champion `seed_sd` > 0.0093* — was read against
the all-mission figure and was not met, and is not met on the TESS figure
either. The run is not falsified. Its estimand is narrower than it appeared.

Filed as [#123](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/123);
the fix is pre-registered in
[`p2-seed-sd-population-2026-09-20.md`](p2-seed-sd-population-2026-09-20.md).
