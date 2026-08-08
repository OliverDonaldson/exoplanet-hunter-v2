# ExoMiner deep dive — full report, 2026-08-07

**Provenance.** This is the verbatim report of a read-only agent audit comparing
NASA's ExoMiner (v2.0, extracted from `ExoMiner-main.zip`) against this
repository. It is reproduced in full and unedited.

**What was independently verified** (by the session that commissioned it): the
repo size figures (124 Python files, ~33k LOC, 30 C/C++ files, zero notebooks);
the existence of `vetting_tce_catalog_exominer_dash_app.py` under
`exominer_vetting_pc_catalog_dash-render-web-app/` with its own `requirements.txt`
and `assets/`. Everything else below — line numbers, code claims, the asserted
bugs in ExoMiner, and the v2 comparisons — is the agent's own analysis and has
**not** been independently checked. Treat specific line references as leads to
confirm, not as established fact.

Paths are relative to the extracted `ExoMiner-main/` for ExoMiner and
`/Users/ollie/Project/v2/` for v2.

---

## Part 1 — The 10 highest-value things v2 should adopt

### 1. Share conv weights across views that are meant to be *compared*
`models/models_keras.py:1507-1768` (`build_joint_local_conv_branches`) and `:1065-1081`.

ExoMiner stacks local flux, odd, even, weak-secondary and all 20 unfolded phases
onto a new axis (`:1247-1278`), runs them through **one** conv tower wrapped in
`TimeDistributed` (`:1533`), then `SplitLayer`s the feature maps back apart
(`:1673-1685`) and gives each branch its own scalars (`:1707-1720`). For odd/even
it then does `Subtract` on the two feature vectors (`:1078-1079`).

This is the single most important architectural idea in the repo. An
odd-vs-even depth comparison is only meaningful if both views pass through
*identical* weights — otherwise a depth difference the head sees is partly a
weights difference. v2 builds `odd_view` and `even_view` as two fully
independent towers (`pipeline/src/exoplanet_hunter/models/cnn_branches.py:198-210`
loops `_branch` per view with `name=name`, so `odd_view_conv0` and
`even_view_conv0` are separate kernels).

**v2 change:** in `cnn_branches.py`, hoist the local-family views (`local_view`,
`odd_view`, `even_view`, `secondary_view`, `unfolded_view` — all 31-bin) into one
`TimeDistributed(_conv_tower)` call, split the output, and feed `odd − even` (not
`odd`, `even`) into fusion. That removes ~4 towers' worth of parameters, ties the
comparison, and gives the head an explicit odd/even *contrast* feature.

### 2. Declarative per-feature normalisation policy, fitted on train, materialised as an artefact
`src_preprocessing/normalize_tfrecord_dataset/config_compute_normalization_stats.yaml:39-...`,
computed by `compute_normalization_stats_tfrecords.py:24-84`, applied by
`normalize_data_tfrecords.py:26-96`.

Every scalar carries a spec:
```yaml
  pgram_max_power:
    clip_factor: 20        # in MAD-std units, applied BEFORE standardising
    log_transform: false
    log_transform_eps: .nan
    missing_value: null    # sentinel that means "absent", excluded from stats
    replace_value: null    # what to impute; null -> train median
    standardize: true
```
Stats are median + `astropy.stats.mad_std` with an explicit fallback to `np.std`
when MAD is exactly zero (`compute_normalization_stats_tfrecords.py:68-70`), and
they are written to `train_scalarparam_norm_stats.npy` **plus** a human-readable
CSV next to the dataset (`:73-83`), then reloaded verbatim at inference
(`exominer_pipeline/run_pipeline.py:150-155`).

v2 has the robust part (`datasets/viewset_pipeline.py:47-59`, median + 1.4826·MAD;
clip at ±10 post-scaling, `:110`) but it is **uniform across all columns** and
lives only as a `ScalarConstants` dataclass inside the joblib bundle. There is no
per-column log transform and no sentinel/imputation policy — `np.nanmedian`
silently drops NaN rather than treating "missing" as a modelled state.

**v2 change:** add a `conf/normalisation.yaml` keyed by the names in
`viewset_tfrecords.FEATURE_COLUMNS`, with `log_transform` for the genuinely
log-scaled DV statistics (`bootstrap_significance` spans 1e-90 to 1 — your own
docstring says so at `viewset_pipeline.py:50-53`, and you currently MAD-scale it
in linear space), and dump the fitted table to CSV beside `cnn_calibrator.joblib`
so a reviewer can read what scale each feature was measured on.

### 3. Paired Wilcoxon signed-rank across matched folds, for comparing two runs
`src_cv/postprocessing/compute_confidence_interval.py:73-99`.

They load `res_eval.npy` for the *same fold index* from two experiment
directories and run `scipy.stats.wilcoxon(exp2_vals, exp1_vals)`. Pairing on fold
index removes fold-difficulty variance from the comparison entirely — which is
exactly the confound behind v2's sd ~0.0106 problem.

**v2 change:** `validation/promotion.py:222-224` currently gates on
`cand_auc <= inc_auc`, a bare comparison of two means each carrying ~0.011 of
noise. Add a paired test over the per-fold rows both summaries already store
(`train_branches.py:279`, `payload["folds"]`), and require the candidate to win
the *paired* comparison, not just the mean. This is the highest-leverage rigour
fix available to you and it costs about 15 lines.

### 4. Cohen's *d* effect size alongside the delta
`src_cv/postprocessing/compute_effect_size.py:11-25` — pooled-std standardised
mean difference between two CV experiments.

**v2 change:** `PromotionDecision.reasons` currently reads
`ROC-AUC 0.9871 vs incumbent 0.9863`. Adding `(d = 0.08, i.e. well inside fold
noise)` would have flagged your stage-2(a) comparison as non-informative before
you spent runs on it.

### 5. Per-example score uncertainty from a super-ensemble, published in the catalogue
`src_cv/postprocessing/combine_predictions_unlabeled_dataset.py:41-85`.

For unlabelled TCEs (which are in no fold's training set), every CV iteration's
ensemble scores every example; they then emit `score_cv_iter_0…N`, `mean_score`,
`std_score` per row. That `std_score` is what appears as **"ExoMiner Unc. Score"**
in the public catalogue — median 0.063, max 0.207 on a [0,1] score across the
11,289 published rows.

v2 has the machinery (`api/app/routes/score.py:182` returns `per_fold`,
`prob_std`; `VettingPanel.tsx:184-192` draws the five per-fold dots) but it is
**live-only**. It is not in `results/candidates_scored.parquet` as a first-class
published column and it is not in the CSV export.

**v2 change:** in `pipeline/scripts/score_candidates.py`, persist
`score_fold_0…4` and `score_std` per candidate, and surface `score_std` as a
sortable column in `CandidatesTable.tsx` + the CSV export from
`api/app/routes/candidates.py`. A reviewer sorting by score descending currently
cannot see which of the top 50 the ensemble disagrees about.

### 6. Provenance headers written *into* every CSV artefact
`src_cv/postprocessing/compute_metrics_cv_run.py:159-168`,
`src/predict/predict_model.py:102-110`,
`src_cv/preprocessing/create_cv_folds_tables.py:36-45`.

The pattern is consistent across the repo:
```python
metrics_df.attrs['CV experiment'] = str(cv_run_dir)
metrics_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
metrics_df.attrs['label_map'] = label_map
with open(fp, "w") as f:
    for k, v in metrics_df.attrs.items():
        f.write(f"# {k}: {v}\n")
    metrics_df.to_csv(f, index=False)
```
and every reader passes `comment='#'`. Fold tables carry the random seed and
split method; predictions carry the label map and experiment name. **The result
is that no output file in the tree is orphaned from the run that made it.**

**v2 change:** apply this to `results/*.csv` and `results/*.parquet`. `results/`
currently holds `candidates_validated.csv`, `candidates_validated_v2.csv`,
`candidates_validated_merged.csv`, `candidates_validated_final.csv` with no
in-file record of which run, threshold, or model version produced each. That is a
provenance hole in a directory you'd hand to an external reader.

### 7. Pin the code version *inside* the model config
`models/exominer_new.yaml:1` — the file's first line is
`commit_id: 52a6d216389a6f2606173c2ad060242738e623fe`, and the whole resolved
config is re-dumped into every run directory as `config_cv.yaml` +
`model_config.yaml` (`src_cv/train/setup_cv_iter.py:34-39`,
`src/train/setup_train.py:29-34`).

v2's `cv_summary.json` `run_config` block (`train_branches.py:301-306`) records
`CVConfig` + view shapes + `n_examples` — genuinely good — but no git SHA and no
architecture config.

**v2 change:** add `git_sha` (from
`subprocess.check_output(["git","rev-parse","HEAD"])`) and the resolved Hydra
`model` group to the `run_config` payload. `registry.json` should carry it too.

### 8. N models per fold with a hard "all models present" gate before ensembling
`src_cv/train/run_cv_iter_modular.sh:90-109` trains `N_MODELS_PER_CV_ITER` models
per CV iteration, `:111-127` scans the directory and **exits 1** if any
`model.keras` is missing, `:141-158` builds the average ensemble, `:161-175`
evaluates and predicts with it. Resumability is built in (`:96-99` skips an
already-trained model).

v2 trains exactly one model per fold (`train_branches.py:147-183`), so the fold's
score *is* a single seed draw — which is precisely why your per-fold noise reads
as sd 0.0106.

**v2 change:** parameterise `CVConfig` with `n_models_per_fold` (start at 3),
average the fold's predictions before calibration, and — critically — **also
write each individual model's metrics**, which gives you a direct within-fold
seed-variance estimate for free (see §Q5 below).

### 9. Serialisable custom layers, registered by name
`models/custom_layers.py:9,21,55,103,154,166,190` — every custom layer and
initialiser carries `@tf.keras.utils.register_keras_serializable()` and a
`get_config()`, plus a belt-and-braces `register_custom_objects()`
(`models/utils_models.py:153-166`).

v2 does this correctly and for a stated reason (`cnn_branches.py:114-129`, the
`PresenceFlag` docstring on why not `Lambda`) — so this is *confirmation*, not a
change. Worth noting because it's one of the few places both repos independently
reached the same conclusion.

### 10. Deployment as a versioned container with the model DOI in the image labels
`Dockerfile:5-10`:
```
org.opencontainers.image.exominer.model="ExoMiner++ (TESS SPOC 2-min S1-S67, Ensemble from CV Iteration 0) | DOI: https://doi.org/10.48550/arXiv.2502.09790"
```
plus
`--exominer_model {exominer++_single | exominer++_cviter-mean-ensemble | exominer++_cv-super-mean-ensemble}`
as a first-class CLI choice (`exominer_pipeline/run_pipeline.py:406-409`).

**v2 change:** put `run_id` and `test_roc_auc_mean` from `models/registry.json`
into `docker/` image labels at build time, and expose the promoted run id in
`GET /` of the API. Right now `model_version` reaches the console
(`score.py:282`) but not the image.

---

## Part 2 — Answers to the six questions

### Q1. Architecture and code organisation

**Layout** (`README.md:38-50`). Five top-level Python trees plus two apps:

| Tree | Role | Size |
|---|---|---|
| `src_preprocessing/` | FITS/DV-XML → TFRecords; TCE tables; ephemeris matching; difference images; normalisation | ~60 files, the bulk |
| `src/` | train / evaluate / predict / postprocess on a fixed split | 16 files |
| `src_cv/` | the same, orchestrated over K folds, + fold construction and metric aggregation | 22 files |
| `src_hpo/` | BOHB via `hpbandster` | 8 files |
| `models/` | architectures + config YAMLs + ensembling | 8 files |
| `exominer_pipeline/` | end-to-end TIC-IDs → scores, containerised | 11 files |
| `exominer_vetting_pc_catalog_dash-render-web-app/` | the public catalogue app | 3 files |

**Module boundaries are clean at the directory level and enforced by nothing.**
There is no `pyproject.toml`, no `setup.py`, no `requirements.txt` at root — only
conda env YAMLs (`others/envs/`,
`exominer_pipeline/conda_env_exoplnt_dl_{amd64,arm64}.yml`). Imports are
absolute-from-repo-root (`from src.utils.utils_dataio import …`) and work only
because `Dockerfile:17` sets `ENV PYTHONPATH="/app"`. The package is not
installable.

**Config flow: YAML + argparse, no dataclasses, and it is layered.**
`src_cv/train/config_cv_train.yaml` holds run-level params (paths, seeds, batch
size, callbacks, label map); `models/exominer_new.yaml` holds the architecture
(`features_set` + `config` blocks). `setup_cv_iter.py:28-32` resolves fold file
paths and merges the model YAML into the run YAML, then `:34-39` snapshots both
into the run directory. Training is then a pure function of one self-contained
YAML (`train_model.py:194-206`). Dispatch is
`getattr(models_keras, config['model_architecture'])` (`train_model.py:114`).

Hyperparameters are **centralised in the model YAML but namespaced by branch**
(`exominer_new.yaml:206-260`: `diff_img_*`, `flux_periodogram_*`, `global_flux_*`,
`local_fluxes_*`, `clf_head_*`). This is good. What is *not* good: the same
concept has different key names in different architecture classes —
`ExoMinerPlusPlus.build_scalar_branches` reads `num_fc_conv_units`
(`models_keras.py:3163`) while `exominer_new.yaml` declares `branch_num_fc_units`
(`:249`). `validate_config` (`train_model.py:26-36`) checks five top-level keys
and nothing else, so a mismatch is a `KeyError` deep inside model construction.

**Preprocessing / training / evaluation separation is genuine and it is enforced
by the filesystem.** Preprocessing writes TFRecords + normalisation stats to
disk; training reads only TFRecords; evaluation and prediction load a saved
`.keras` and re-read the same TFRecords. Each stage is a separate process invoked
by a shell script (`run_cv_iter_modular.sh`). Coupling is by file path in YAML.

**Worth copying:** the config-snapshot-into-run-dir pattern; branch-namespaced
hyperparameter keys; the layered run-config/model-config split; provenance
headers in CSVs; the strict stage separation via on-disk artefacts.

**Just their history:** `src/` vs `src_cv/` duplicate a lot (both have `train/`,
`predict/`, `postprocessing/`); `ExoMinerMLP`/`ExoMinerSmall`/`ExoMinerDiffImg`/
`ExoMinerJointLocalFlux`/`ExoMinerPlusPlus`/`ExoMinerPlusPlusTemp` are six
near-copies of the same 900-line build sequence in one 4,551-line file;
`src_preprocessing/light_curve/` and `src_preprocessing/fast_ops/` are duplicated
Google code (see Q2); `others/` is a licence-and-media dumping ground.

### Q2. The C/C++ component — it is dead code, twice

30 C/C++/CLIF/BUILD files, in **two identical copies**:
- `src_preprocessing/fast_ops/`
- `src_preprocessing/light_curve/fast_ops/`

`diff -rq` between them shows the only difference is that the first copy also has
three `*_test.py` files.

**What they are:** verbatim Google `exoplanet-ml` / AstroNet code by Chris
Shallue, 2018. The include guard is
`TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_FAST_OPS_VIEW_GENERATOR_H_`
(`fast_ops/view_generator.h:16`), the namespace is `astronet` (`:22`), and the
CLIF file imports from a path that does not exist here:
`from "third_party/tensorflow_models/astronet/light_curve/fast_ops/view_generator.h"`
(`fast_ops/python/view_generator.clif:20`). `light_curve/README.md:3-4` credits
`@cshallue` directly.

**Why they exist:** phase-folding + median-filtering a light curve into a
fixed-bin "view" is the inner loop of preprocessing, and `ViewGenerator`
(`view_generator.h:31-36`) keeps the phase-folded curve in C++ object state "to
minimize expensive copies between the language barrier" — i.e. fold once, then
generate global (301-bin) and local (31-bin) views without re-folding or
re-crossing into Python.

**Are they called?** No. Grep across the whole repo finds `fast_ops` referenced
only from its own three test files and one README line. There is **no
`WORKSPACE`, no `MODULE.bazel`, no `.bazelrc`** anywhere in the tree, so the
`BUILD` files cannot be executed; there is no `__init__.py` in `fast_ops/python/`,
so `from light_curve.fast_ops.python import median_filter`
(`fast_ops/python/median_filter_test.py:24`) cannot resolve. Meanwhile the live
preprocessing path uses the **pure-Python** sibling:
`src_preprocessing/lc_preprocessing/phase_fold_and_binning.py:10` imports
`from src_preprocessing.light_curve import median_filter, util`, which is
`light_curve/median_filter.py`, not the C++ one.

**Verdict: legacy artefact, not a performance technique to adopt.** It is
inherited AstroNet lineage that was never wired up in this repo and then got
copy-pasted a second time. There is nothing here for v2 to steal — v2's binning
is NumPy in `preprocess/views.py`, which is the same choice ExoMiner actually
made in practice. If v2's binning ever becomes the bottleneck the answer is Numba
or a vectorised `np.add.at`, not Bazel + CLIF.

### Q3. Model and training practice, vs v2

**Model definition style.** Class-per-architecture wrapping a Keras Functional
graph: `__init__` builds inputs from the feature spec (`utils_models.py:14-45`),
calls `self.build()`, and exposes `.kerasModel` (`models_keras.py:2387-2392`).
`build()` composes `build_scalar_branches()` + `build_conv_branches()` +
`build_joint_local_conv_branches()` + `build_conv_unfolded_flux()` +
`build_diff_img_branch()` into a dict, `connect_segments()` concatenates the dict
values (`:3191-3208`), `build_fc_block()` adds the head (`:3210-3262`), and a
`Dense(output_size)` + sigmoid/softmax closes it (`:3292-3297`).

Branch topology is **entirely data-driven from YAML** (`exominer_new.yaml:100-188`):
each branch names its `views` and its `scalars`, and the builder loops over
`self.config['conv_branches']`. Adding a diagnostic is a YAML edit plus a feature
in the TFRecords.

v2 hardcodes the equivalent as module-level dicts — `BRANCH_SCALARS` and
`SCALAR_BRANCHES` (`cnn_branches.py:31-58`) — with `VIEW_SHAPES` imported from
`viewset_io`. That is arguably *better* for a smaller project (type-checked,
greppable, no YAML-key typos), but it means branch topology is not part of the
run config and therefore not in `cv_summary.json`.

**Where v2 is structurally ahead:** presence gating. `_gated()`
(`cnn_branches.py:150-157`) multiplies each branch embedding by a `PresenceFlag`
so a branch with no measured bins contributes exactly zero, and the mask vector
rides into the head (`:220`). ExoMiner's nearest equivalent is quality-weighted
fusion for difference images only (`models_keras.py:1150-1179`, multiply features
by the per-sector quality metric then `ReduceSum`); the flux branches have no
missingness concept at all — they impute in preprocessing (`utils_imputing.py`)
and the model never learns that a value was imputed.

**Input pipeline.** `src/utils/utils_dataio.py:150-493`, `InputFnv2`. Notable
choices:
- Filename-level shuffle before record-level shuffle, TRAIN only (`:440-441`).
- `interleave` with `cycle_length=AUTOTUNE`, 64 MB read buffer per file, forced to
  `1` in PREDICT mode (`:446-450`) — **deterministic ordering for inference**,
  parallel for training. `map` and `batch` likewise take `deterministic=True` only
  in PREDICT (`:477-478`, `:484-486`).
- `tf.lookup.StaticHashTable` for label→id (`:278-295`) and for category→weight
  (`:297-315`), built once in `__init__`.
- **`tf.debugging.check_numerics` on every parsed feature** (`:352`) and on the
  label (`:378`), plus `assert_greater_equal(label_id, 0)` (`:381`) so an unmapped
  label fails loudly instead of training as class −1.
- Augmentation params drawn once per example and applied consistently across all
  `*view*` features (`:387-405` + `prepare_augment_example_online:47-78`) —
  reverse and bin-shift are shared, so odd/even/secondary stay in phase with each
  other.
- No `cache()` (commented out at `:452-456`), `prefetch(AUTOTUNE)`.

v2's `make_viewset_dataset` (`viewset_pipeline.py:62-129`) is smaller and does the
two things that matter: split membership via `StaticHashTable` on `tic_id`
(`:89-92`) and `cache()` before augment so augmentation redraws each epoch
(`:120-126`). It does not have the numerics assertions. **Adopt `check_numerics` +
the label assertion** — cheap, and it converts a silent NaN-poisoned epoch into a
traceback.

**CV protocol.** ExoMiner: fold tables are built once, offline, at the
*target-star* level with a greedy planet-count balance
(`create_cv_folds_tables.py:81-106`) — a hand-rolled StratifiedGroupKFold. Then
each fold's normalised copy of the dataset is materialised on disk with
statistics computed on **that fold's training shards only**
(`preprocess_cv_folds_trecord_dataset.py:106`,
`compute_normalization_stats(data_shards_fps['train'], …)`). Then N models are
trained per fold and averaged.

v2 uses `StratifiedGroupKFold(shuffle=True, random_state=seed)` grouped on
`tic_id` with an inner `GroupShuffleSplit` for validation
(`train_branches.py:244-254`). **This is cleaner than ExoMiner's greedy
heuristic** — and, as noted below, ExoMiner's heuristic has a bug.

**Leakage guards, head to head:**

| Guard | ExoMiner | v2 |
|---|---|---|
| Group by star | ✅ `split_tces_by_target*` | ✅ `groups=tic_id` |
| Norm stats from train only | ✅ per fold, materialised (`preprocess_cv_folds_trecord_dataset.py:106`) | ✅ `fit_scalar_constants(index.iloc[train_idx], …)` (`train_branches.py:131`) |
| Augment train only | ✅ `data_augmentation and mode=='TRAIN'` (`utils_dataio.py:198`) | ✅ `augment if split is TRAIN else None` (`:141-144`) |
| Calibration fit on val only | — (no calibration at all) | ✅ `PlattScaler.from_validation` (`:192`) |
| Score the reloaded checkpoint | — (`model.save` then separate eval process, so effectively yes) | ✅ explicit, with a documented 0.31/example drift (`:184-188`) |
| Prediction/index alignment asserted | — | ✅ hard `RuntimeError` (`:217-218`) |

**Calibration: ExoMiner has none.** The Dash app's own tooltip says so — *"It is
NOT a probability"* (`vetting_tce_catalog_exominer_dash_app.py:80-81`). No Platt,
no isotonic, no Brier, no ECE anywhere in the repo. v2 is unambiguously ahead
here.

**Ensembling.** ExoMiner does it at three levels: (a) N models per fold, averaged
into a saved `ensemble_avg_model.keras` — a real Keras graph with an `Average`
layer, not a post-hoc mean (`utils_models.py:48-68`,
`create_ensemble_avg_model.py:19-45`); (b) mean across CV-iteration ensembles for
unlabelled data (`combine_predictions_unlabeled_dataset.py:82`); (c)
`exominer++_cv-super-mean-ensemble` as a shipping model choice. v2 does (b) only,
with one model per fold.

Baking the ensemble into a single `.keras` artefact is worth copying — it makes
the served object one file with one load path, rather than five checkpoints plus
averaging logic in `scoring/ensemble.py`.

### Q4. `vetting_tce_catalog_exominer_dash_app.py` — what it actually is

**270 lines. One file. It is a static table browser, not a vetting console.**

Concretely:
- Reads one 2.8 MB CSV at import time (`:24-36`), 11,289 rows, 11 of the 16
  columns used. No database, no API, no model, no live inference.
- Renders a `dash_table.DataTable` with `filter_action="native"`,
  `sort_action="native"`, `page_size=30` (`:155-198`).
- Header tooltips explaining each column (`:66-84`) — including the honest
  *"model score in [0,1] … It is NOT a probability"*.
- One custom regex filter on Sector Run (`:216-236`).
- Export-visible-rows-to-CSV button (`:239-265`), using `derived_virtual_data` so
  the export respects the current filter/sort.
- A `DV mini-report URL` column rendered as markdown, each row deep-linking to
  that TCE's SPOC DV mini-report PDF on MAST (`:57`, `:193-195`).
- Provenance in the page header: *"Results from 1/16/2025 10:14am | Last web app
  update: 10/28/2025 1:01pm | Excluded TCEs with scores < 0.1"* (`:122-135`), plus
  a citation request and a Zenodo DOI (`:107-149`).
- Deployed on Render (the directory name says so) as a Flask/gunicorn app —
  `server = app.server` (`:43`), `requirements.txt` pins `dash==3.0.4`,
  `Flask==3.0.3`, `gunicorn==23.0.0`.

**There are no plots. No light curves, no phase folds, no odd/even overlay, no
centroid track, no periodogram. No per-TCE detail view at all.** The user clicks a
link and downloads a PDF produced by the SPOC pipeline, not by ExoMiner.

**Where ExoMiner is genuinely ahead of v2:**

1. **Per-row score uncertainty is published.** `ExoMiner Unc. Score` is a real
   column (median 0.063, max 0.207, non-zero for 99.9% of rows). v2 computes this
   but does not persist or publish it. See adoption item #5.
2. **Scale and completeness.** 11,289 TCEs covering all of TESS SPOC 2-min
   S1–S67, with 4,256 scoring above 0.9. This is a survey product. v2's
   `results/candidates_scored.parquet` is a shortlist.
3. **Deference to the authoritative artefact.** Rather than reimplementing SPOC's
   diagnostics, they link to the SPOC DV mini-report — the document a professional
   vetter actually reads. That is a legitimate design choice, not a cop-out.
4. **Citation, DOI, dated provenance in the UI.** The header tells you exactly
   which run the numbers came from and asks to be cited. v2's console has nothing
   equivalent.
5. **Zero operational surface.** No model in the process, no cold start, no lock.
   It cannot 503.

**Where v2 is genuinely ahead — and it's not close:**

v2 does live inference on demand. `GET /score/{tic_id}`
(`api/app/routes/score.py:108-289`) fetches the light curve, cleans it, resolves
an ephemeris (user > catalogue > BLS), builds views, runs the promoted 5-fold
ensemble with MC-Dropout, and returns a **calibrated** probability plus per-fold
spread plus eight diagnostic blocks. `VettingPanel.tsx` renders:
- a probability bar with the ±MC-dropout band, the five per-fold dots, and the
  decision threshold marked (`:158-195`);
- global and local phase views (`:299-300`);
- **odd vs even on shared axes** (`:63-120`, `:301-303`) — the single most
  diagnostic plot for an EB, which ExoMiner's app does not have;
- centroid offset track (`:304-311`);
- an on-demand BLS periodogram with the best period marked (`:122-156`,
  `:312-322`);
- σ-level readouts with `warn` styling for centroid shift, odd/even depth *and
  timing*, secondary eclipse with false-alarm threshold and albedo, duration
  consistency (`q/q_circ`, `a/R*`), and BLS false-alarm checks — SWEET, asymmetry,
  depth mean/median, gap fraction (`:323-397`).

Plus `/reliability` (calibration curve), a filterable/sortable/exportable
candidate catalogue (`routes/candidates.py`), and the whole TRICERATOPS FPP/NFPP
statistical-validation path (`validation/statistical.py`).

**The honest summary:** the v2 owner's belief was *directionally* right and
*specifically* wrong. ExoMiner does have a public interactive artefact — it's just
a catalogue table, and v2's console does strictly more per candidate. What
ExoMiner has that v2 doesn't is a **published survey-scale catalogue with per-row
uncertainty, a DOI, and a citation ask.** That's a publishing gap, not an
engineering gap.

### Q5. Evaluation and reporting — and the run-variance question

**Metrics.** `src/postprocessing/compute_metrics_from_predictions_csv_file.py:30-179`
computes, from a predictions CSV rather than from a live model:
- Ranking: `auc_pr` (1000 thresholds, interpolated), `auc_roc`, `avg_precision`.
- Operating point at a threshold: precision, recall, accuracy, balanced accuracy,
  F1.
- **Threshold-free operating-point metrics**: `PrecisionAtRecall(0.99)` and
  `RecallAtPrecision(0.99)` (`:102-103`, thresholds set at
  `compute_metrics_cv_run.py:176-177`).
- **Precision-at-k for k ∈ {50, 100, 150, 200, 500, 1000, 2000, 3000}**
  (`:161-167`) — this is the shortlist metric for a survey, and it's the right one.
- **Per-category recall and counts** — not just per class-id. `recall_EB`,
  `recall_NTP`, `recall_BD`, `n_EB`, … (`:155-159`). So you can see that the model
  catches confirmed planets but leaks brown dwarfs.

**Per-population slicing** is by disposition category (above) and by `obs_type`
(2-min vs FFI) via a filter on the predictions table (`:236`, commented) plus a
dedicated `ComputePerformanceOnFFIand2min` callback (referenced but commented out
at `src/train/train_model.py:92-93`).

**Aggregation.** `compute_metrics_stats_cv_run` (`compute_metrics_cv_run.py:14-168`)
reads each fold's predictions CSV, computes the full metric block per fold,
appends `mean` and `std` rows across folds (`:114-124`), and *additionally*
computes metrics on the **pooled out-of-fold predictions across all folds**
(`:131-152`) — with a shouted caveat,
`# ONLY VALID FOR NON-OVERLAPPING CV ITERATIONS' SETS!!!` (`:133`). Everything
lands in `metrics_allfolds_with_stats.csv` with the metadata header.

**Uncertainty / error bars.** Two mechanisms:
- `compute_confidence_interval.py:13-41` — t-interval on the mean across folds,
  `sem = std/√n`, `df = n−1`.
- `compute_effect_size.py:11-25` — Cohen's *d* between two experiments' fold means.
- `compute_confidence_interval.py:73-99` — paired Wilcoxon signed-rank on matched
  folds.

**Now, directly on v2's discovery.**

**ExoMiner has exactly the same conflation, one level up.** Their `std`
(`compute_metrics_cv_run.py:122`) is the spread of the *fold-level* metric — which
mixes fold difficulty with training noise, and is then fed into a t-interval
(`compute_confidence_interval.py:36-39`) as if folds were i.i.d. draws. They do not
report seed variance at a fixed fold anywhere. So if you were hoping ExoMiner
solved this: they didn't.

**But their setup is better positioned than v2's, for two reasons, and one of
them you can copy today.**

*First,* the thing whose spread they report is an **ensemble of N models**, not a
single seed draw. The seed noise is averaged down by √N before the across-fold std
is taken. v2's fold metric is one model, so v2's 0.0106 contains the full per-seed
variance. That is why item #8 (N models per fold) matters — it shrinks the number
you're reporting rather than just relabelling it.

*Second,* **the data to decompose the variance already exists in their artefact
tree and they simply don't use it.** Every model writes its own `res_eval.npy`
(`evaluate_model.py:98`) into `cv_iter_K/models/modelI/`, and
`compute_metrics_stats_cv_run` takes a `results_sub_dir` argument that can be
pointed at `'models/model0'` instead of `'ensemble_model'`
(`compute_metrics_cv_run.py:15`, and the commented alternative at `:211`). Run it
once per model index and you get an N×K matrix of AUCs — from which within-fold
(seed) and between-fold (difficulty) variance separate cleanly.

**The v2 change this implies is concrete:** train n≥3 models per fold, write
per-model fold metrics into `cv_summary.json` as `folds[k]["models"][i]`, and
report **two** numbers instead of one — `sd_seed` (pooled within-fold sd across
models) and `sd_fold` (sd of fold means). Your current ±0.0106 is
`sqrt(sd_seed² + sd_fold²)` with no way to tell which dominates. Then gate
promotion on the paired-fold Wilcoxon (item #3), which is invariant to `sd_fold`
entirely.

One more thing worth stealing: `RecallAtPrecision(0.99)` /
`PrecisionAtRecall(0.99)` and precision-at-k. v2 has `recall_at_1pct_fpr`
(`eval/comparison.py:35-50`, correctly read off the ROC curve rather than by
counting, which is the better implementation) but not precision-at-k. For a survey
shortlist where the reviewer will look at exactly the top 50, precision@50 is the
number that matters and neither AUC nor recall@FPR reports it.

**How they present results in docs.** Thinly. `docs/` is four files, all about
*running the Podman pipeline* — no results, no metrics, no model cards.
`docs/exominer-features.md` is a bare feature/dim/dtype table. All actual
performance reporting lives in the AJ papers (`README.md:87-91`) and the Zenodo
deposit, not in the repo. **v2's `docs/` is considerably better**
(`architecture.md`, `data_provenance.md`, `roadmap.md`, `OPERATING.md`,
`DEPLOY.md`, `figures/`).

### Q6. Engineering practices

| | ExoMiner | v2 |
|---|---|---|
| **CI** | none — no `.github/`, no `.gitlab-ci.yml`, nothing | `.github/workflows/ci.yml`: 3+ jobs, ruff, pytest with `-m "not network and not slow"`, API contract tests, frontend build |
| **Tests** | 7 files. 3 are unrunnable Google CLIF tests; `models/test_model_architecture.py` is a script with a hardcoded `/u/msaragoc/…` path and no assertions; `src/utils/test_input_fn.py` likewise. Effectively **one** real test file (`src_cv/preprocessing/test_create_iterations_list_for_cv.py`, 176 lines) | 32 test files under `pipeline/tests/` + 4 under `api/tests/`, with `conftest.py` and marker-based selection |
| **Lint / types** | none | ruff + ruff-format + mypy on `pipeline/src`, all in `.pre-commit-config.yaml`, with `exclude: ^pipeline/vendor/` so vendored source stays diffable |
| **Packaging** | none installable; `PYTHONPATH` only | `pip install -e ./pipeline[dev] -e ./api[dev]`, extras (`[validation]` for TRICERATOPS) |
| **Deps** | conda YAMLs per arch, no lockfile, no root requirements | `environment.yml` + per-package pins; Dash app in ExoMiner *does* have a fully pinned `requirements.txt` (29 packages) — the one place they got it right |
| **Seeding** | **`tf.random.set_seed` appears zero times in the repo.** No `np.random.seed`, no `PYTHONHASHSEED`, no `TF_DETERMINISTIC_OPS`. Only NumPy `default_rng(seed)` for data splitting and imputation | `utils/seeds.py:9-34` seeds Python/NumPy/TF/Keras, called from 8 entry points |
| **Artefact versioning** | `commit_id` in the model YAML; full config snapshot per run dir; provenance headers in CSVs; OCI labels with model DOI | `registry.json` + DVC + MLflow + `run_config` in `cv_summary.json` |
| **Reproducibility of a run** | training is a pure function of one snapshotted YAML — genuinely good — but unseeded, so not bit-reproducible | seeded, and the checkpoint is reloaded before scoring |
| **Hardcoded paths** | **258 occurrences of `/u/msaragoc`, `/nobackup`, or `/Users/msaragoc` across 57 files** — including every `__main__` block, so most scripts are unrunnable as shipped | `ProjectPaths.from_cfg`, env-var overrides in the API |
| **Config loading** | `yaml.unsafe_load` in **17 files, 22 call sites** | Hydra + `yaml.safe_load` |

**What actually makes ExoMiner credible to an outside reader**, despite all of the
above: the Podman one-liner with a published `ghcr.io` image, the GIF demos in the
README, the four-page runnable docs, the model DOI in the image labels, the Zenodo
data deposit, the live catalogue app, and three peer-reviewed AJ papers.
**Distribution and citation, not code hygiene.** That's the actual lesson — v2's
engineering is better and its distribution story is weaker.

---

## Part 3 — Where v2 is already equal or better (evidence-based)

1. **Calibration.** ExoMiner has none and says so in its own tooltip
   (`dash_app.py:80-81`). v2 has Platt scaling fitted on validation only, per fold,
   persisted in the bundle (`train_branches.py:192-205`), plus Brier and ECE as
   first-class metrics and an ECE promotion guard (`promotion.py:229-233`). Not
   close.

2. **Operating-point vs ranking metrics.** `SliceMetrics` carries
   `recall_at_{1,5,10}pct_fpr` alongside `roc_auc`/`pr_auc`/`brier`/`ece`
   (`comparison.py:53-89`), and `recall_at_fpr` reads off the ROC curve rather than
   counting negatives, with the tie-handling reason documented (`:41-42`).
   ExoMiner's `precision_at_k` is a `keras.metrics.Precision(top_k=k)`
   (`compute_metrics_from_predictions_csv_file.py:162`), which is coarser.

3. **Population accounting.** `mission_coverage` / `MissionCoverage.dropped`
   (`comparison.py:92-161`) exists specifically because an inner join silently
   dropped all 527 K2 rows. ExoMiner has no equivalent —
   `compute_metrics_stats_cv_run` compares whatever is in each directory with no
   check that the rows match.

4. **The promotion gate.** `evaluate_promotion` (`promotion.py:132-246`) is
   materially more rigorous than anything in ExoMiner: NaN-first checking
   (`:187-200`, with the reasoning that NaN loses every inequality and would sail
   through), an explicit gate population with the reason the pooled aggregate is
   unfit (`:26-33`), a population-mismatch refusal (`:207-220`), and separate
   Brier/ECE/recall tolerances. ExoMiner has **no promotion concept at all** — the
   model that ships is chosen by hand and baked into a Dockerfile label.

5. **Testing and CI.** 36 test files with markers and a working three-job CI,
   versus one genuinely runnable test file and no CI.

6. **Portability.** 258 hardcoded personal absolute paths mean most ExoMiner
   scripts cannot be run by anyone outside NASA ARC without editing. v2 is
   `ProjectPaths` + env vars + Docker Compose throughout.

7. **The vetting console.** Eight diagnostic blocks with σ-level readouts, five
   plot types, live scoring with MC-Dropout — versus a sortable table. See Q4.

8. **Presence/missingness as a modelled signal.** `PresenceFlag` gating
   (`cnn_branches.py:114-157`) and the mask vector into the head (`:220`) are
   conceptually ahead of ExoMiner's impute-and-forget.

9. **Statistical validation.** `validation/statistical.py` (441 lines) integrates
   TRICERATOPS FPP/NFPP with a degenerate-posterior detector the upstream fork
   lacks (`:66-82`), correct ppm→fraction handling with the failure mode documented
   (`:382-387`), and real SPOC apertures instead of the 5×5 default (`:289-317`).
   ExoMiner has nothing comparable — it produces a score and defers validation to
   the SPOC DV report and to human follow-up.

10. **Prose quality in the code.** v2's module docstrings state *why* with dated
    evidence ("measured drift up to 0.31 per example on cebb0fe6", "TIC 441804533
    came back at exactly 1/21 across 21 scenarios"). ExoMiner's are accurate but
    generic. For an outside reader auditing decisions, v2's are far more useful.

---

## Part 4 — In ExoMiner and should NOT be copied

1. **`yaml.unsafe_load` in 17 files / 22 call sites** (`train_model.py:195`,
   `evaluate_model.py:119`, `predict_model.py:126`, `setup_cv_iter.py:31`,
   `create_ensemble_avg_model.py:66`, …). Arbitrary Python object construction on
   config load. They use it because they round-trip `pathlib.Path` and `tf.DType`
   objects through YAML — which is itself the mistake. Serialise paths as strings.

2. **No seeding of TF/Keras anywhere.** `tf.random.set_seed` count: zero. Combined
   with `shuffle_seed=24` in the data pipeline, a rerun of the same config gives the
   same data order and different weights, so "the same run" is not reproducible. v2
   already handles this — do not regress toward it.

3. **A 4,551-line `models_keras.py` with six near-duplicate architecture classes.**
   `ExoMinerJointLocalFlux` (`:1182`) and `ExoMinerPlusPlus` (`:2361`) and
   `ExoMinerPlusPlusTemp` (`:3301`) share ~85% of their build logic by copy-paste.
   When `use_skip_connection_conv_block` was added it had to be added three times.
   v2's single `build_cnn_branches` + `cnn_dualview` is the right shape; keep it.

4. **Module-level globals leaking into functions — an actual bug.**
   `create_cv_folds_tables.py:37-39` reads `dataset_tbl_fp`, `rnd_seed`, `split_by`
   inside `create_table_folds_statistics()` where none is a parameter, and `:203`
   reads `n_folds_predict` inside `create_cv_folds_tables()` where it is likewise
   not a parameter. All four are defined only in `__main__` (`:216-225`). Importing
   and calling these functions as a library raises `NameError`.

5. **A subtler bug in the default fold splitter.** `create_cv_folds_tables`
   shuffles target-star groups with the seeded RNG (`:151-154`), then
   `split_tces_by_target_balanced` immediately does `tce_tbl.groupby('target_id')`
   (`:93`), which re-sorts by key — pandas `groupby` defaults to `sort=True`. The
   greedy assignment at `:101-105` therefore walks targets in ascending `target_id`
   order regardless of `rnd_seed`. **The seed has no effect on fold composition
   under the default `split_by='target greedy balanced'`**, and the seed is
   nevertheless recorded in the fold-statistics metadata (`:38`) as if it did. v2's
   `StratifiedGroupKFold(shuffle=True, random_state=seed)` is correct; don't replace
   it with a hand-rolled greedy pass.

6. **`.numpy()` inside a `tf.data.map` function.** `utils_dataio.py:412`:
   `output[final_name.numpy().decode()] = feature_value`. This only executes when
   `feature_map is not None`; the shipped `exominer_new.yaml:3` sets
   `feature_map: null`, so the path is dead — but it is latent-broken, since a
   graph-mode `map` has no `.numpy()`.

7. **`__main__` blocks as the configuration surface.** Every postprocessing script
   hardcodes experiment paths in `if __name__ == '__main__'`
   (`compute_metrics_cv_run.py:215-217`, `compute_effect_size.py:32`,
   `combine_predictions_unlabeled_dataset.py:89`,
   `plot_loss_and_metric_curves.py:19`). These aren't scripts, they're notebooks
   with `.py` extensions — you edit the constants and re-run. v2's Hydra + argparse
   is right.

8. **Duplicated vendored trees.** `src_preprocessing/fast_ops/` and
   `src_preprocessing/light_curve/fast_ops/` are byte-identical copies of dead
   Google code. v2's `pipeline/vendor/triceratops/` with the
   `exclude: ^pipeline/vendor/` pre-commit rule and a stated "byte-for-byte as
   upstream ships it" policy is the correct pattern.

9. **Dead code left in-tree rather than in history.** `models_keras.py` has ~600
   lines of commented-out layers; `utils_metrics.py:148-382` is 234 lines of
   commented TF1 metrics; `.gitignore` lists 10 paths under `others/envs/others/envs/`
   that don't exist. It is a working scientist's repo published as-is, and the
   noise-to-signal cost to an outside reader is high.

10. **Fold `std` fed into a t-interval as if folds were i.i.d.**
    (`compute_confidence_interval.py:36-39`). This is the same error v2 just found
    in itself. Copy their paired Wilcoxon (`:73-99`) and their Cohen's *d*; do
    **not** copy the t-interval on across-fold spread.
