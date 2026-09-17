# Scripts

Hydra entry points. Anything reusable lives in
`src/exoplanet_hunter/`; these are thin CLIs over it, so a script is never the
only place a piece of logic exists. That rule exists because it was broken
twice: two scripts once carried their own copy of the aux vector and the
catalogue request, and both silently wrote wrong data
(`preprocess_only.py`, deleted in stage 1, the old stage 0).

Hydra style is `key=value`, **not** `--flag`:

```bash
python pipeline/scripts/score_candidates.py limit_mission=TESS max_candidates=100
```

Three scripts use argparse instead (`validate_data.py`, `promotion_gate.py`,
`validate_candidates.py`) because they are called from CI and the DAG, where
`--flag` is the convention. Check the docstring before assuming.

## Building the dataset

| script | what it does |
|---|---|
| `refresh_labels.py` | rebuild the labelled catalogue only — stage 1 of the build, no downloads |
| `ingest_exofop.py` | build the candidate catalogue from ExoFOP TOI + CTOI exports |
| `build_dataset.py` | the full build: catalogue → download → preprocess → `views.npz` |
| `shard_views.py` | `views.npz` → TFRecord shards for training |
| `fetch_dv.py` | TESS DV report XML for every target the pipeline knows about |
| `build_dv_table.py` | the fetched DV archive → one parquet row per target; builds `dv_usable` |
| `fetch_ffi.py` | FFI light curves for candidates with no 2-minute SPOC product |
| `fetch_momentum_dumps.py` | reaction-wheel desaturation times, per sector |
| `fetch_ruwe.py` | Gaia DR3 RUWE for the pipeline's TESS targets |
| `build_viewset.py` | the branch-model view set for every target in a catalogue |
| `shard_viewset.py` | a built view set → TFRecord shards for the branch trainer |
| `build_synthetic_negatives.py` | stage 8's synthetic-negative view set — arm N |
| `build_fold_assignment.py` | one outer-CV partition, shared by both trainers |

`build_dataset.py` is the **only** preprocessing path. With a warm FITS cache
its download stage is all cache hits.

## Gates

| script | what it does |
|---|---|
| `validate_data.py` | the five validation gates (schemas, views, shrink, leakage) |
| `promotion_gate.py` | does a fresh CV run replace the champion? exit 0 = promote |
| `check_showcase_ready.py` | is this repository fit to put in front of a stranger? (`make ready`) |

Both run in CI against synthetic fixtures and in the DAG against real
artefacts — the same code either way.

## Scoring and evaluation

| script | what it does |
|---|---|
| `score_candidates.py` | bulk-score held-out candidates with the promoted ensemble |
| `validate_candidates.py` | TRICERATOPS FPP/NFPP on the shortlist (needs `pipeline[validation]`) |
| `render_vetting.py` | six-panel vetting figures for top-K scored candidates |
| `injection_recovery.py` | completeness vs S/N through the served path — with the control arm |
| `eval_since_confirmed.py` | prospective eval on candidates whose labels arrived later |
| `uncertainty_eval.py` | does MC-Dropout `prob_std` predict errors? (verdict: no — use distance to threshold) |
| `export_predictions.py` | backfill per-example CV predictions for an already-trained run |
| `recalibrate_run.py` | refit an existing run's calibration bundles in place, no retraining |
| `evaluate.py` | score a run onto a shard set, compare two prediction sets, or summarise one |
| `control_arm.py` | offline control arm for a CV run directory — stage 7i |
| `control_lane.py` | re-score the served model on the current population, so a delta means the model |
| `candidate_bias.py` | observation bias on the candidate population, where the original finding lives |
| `power_analysis.py` | what effect is detectable at this n, which metric gates, and what carries a contrast |
| `export_training_history.py` | a run's per-epoch history out of MLflow into `models/history/` |

## Training

| script | what it does |
|---|---|
| `train_branches.py` | cross-validate the per-diagnostic branch model over the view-set shards |

The dual-view trainer has no script: it is `python -m exoplanet_hunter.training.train`,
a Hydra entry point in the library. `train_branches.py` is a script because it
predates that and takes argparse flags the Hydra config has no key for.

## Figures

| script | what it does |
|---|---|
| `make_performance_figures.py` | the promoted run's curves into `docs/figures/` |
| `plot_provenance.py` | provenance sky maps from the live artefacts |

## Long runs

Score, injection-recovery and validation runs take tens of minutes to hours.
Run them detached, from a real terminal, writing to a plain file — not through
a pipe that dies with the session:

```bash
nohup caffeinate -dis /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/score_candidates.py limit_mission=TESS \
  >> outputs/score-candidates.log 2>&1 &
```

`score_candidates.py` checkpoints every 25 rows and resumes from its output
parquet. `validate_candidates.py` writes its CSV after every target but does
**not** skip completed ones on a re-run.
