# Exoplanet Hunter — Documentation

A self-refreshing, self-validating transit-vetting platform: catalogue refresh →
validation gates → tf.data pipeline → calibrated cross-validated CNN ensemble →
promotion gate → live FastAPI scoring → the vetting console.

## Documents

| Document | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | environment, everyday commands, the refresh loop, scoring a target, deploying |
| [model_specs.md](model_specs.md) | every model input, where it comes from, and the two architectures |
| [model_pipeline.md](model_pipeline.md) | the end-to-end path, stage by stage |
| [PLAN.md](PLAN.md) | where the project stands and the delivery steps left |
| [../CLAUDE.md](../CLAUDE.md) | the rules every session works under |
| [experiments/](experiments/README.md) | the record of what was measured, one frozen file per stage |
| [known-limits.md](known-limits.md) | the weakness register and the limits every result is read under |
| [decisions.md](decisions.md) | decisions taken and open, and what was considered and deferred |
| [roadmap.md](roadmap.md) | the index: every old section number, pointing at where its text lives |
| [overview.md](overview.md) | what the pipeline outputs, column by column, and its known limitations |
| [troubleshooting.md](troubleshooting.md) | failure modes, operational traps, what is safe to delete |
| [data_provenance.md](data_provenance.md) | catalogue sources, sky coverage, and the measured-findings ledger |
| [references.bib](references.bib) | bibliography |
| [figures/](figures) | generated performance and provenance figures |

Start with **getting-started** to run something, **overview** to understand what
comes out, and **PLAN** for where the project stands.

## Code map

Most modules carry their own README with the detail; `docker/` and `infra/`
carry theirs in the files themselves.

| Path | What lives there |
|---|---|
| `pipeline/` | the science: ingest, preprocess, features, training, eval, validation |
| `pipeline/src/exoplanet_hunter/` | the installable library |
| `pipeline/scripts/` | entry points: build, score, gate, figures |
| `pipeline/vendor/` | third-party source patched and shipped |
| `orchestration/` | the Prefect refresh DAG and its schedule |
| `api/` | FastAPI serving |
| `frontend/` | the vetting console: `design-console/`, built by `build.py` into one static file |
| `docker/`, `infra/` | images and deployment config |

## The rules that do not bend

1. **Fresh data only.** Raw FITS are an evictable cache of immutable archive
   files; every derived artefact is rebuilt by current code and versioned in DVC.
2. **The `/score/{tic_id}` contract is pinned.** `api/app/schemas.py` and the
   console's client, `frontend/design-console/src/app.api.js`, change together
   or not at all.
3. **Models ship only through the promotion gate.** Beat the champion's
   out-of-fold ROC-AUC on TESS, without Brier degrading by more than 0.005 or
   ECE by more than 0.01, and without recall @1% FPR falling by more than the
   run's own measured floor, or it does not become the champion. A margin
   inside the floor is UNRESOLVED, a stop-and-ask, not a rejection. The gate
   has correctly rejected several retrains; that is it working.
4. **A margin smaller than its noise floor is not a result.** Every run with
   more than one member per fold measures its own floor, by the rule
   `2 x sd / sqrt(n_models_per_fold)`. A single-member run, including the
   served champion, reports the floor from the calibration run that measured
   it and says so. A floor belongs to the architecture it was measured on.
5. **Pre-registration is binding.** A result landing outside the terms fixed
   before it was run is reported as falsified, never re-specified.
6. **Verify by executing.** Documentation claiming something works is a
   hypothesis. Run the path, check the artefact.

## Glossary

- **Vetting, not detection.** The pipeline classifies a signal another pipeline
  already found (a TOI from TESS, a KOI from Kepler); it does not search light
  curves for new ones.
- **Champion.** The served run, named in `models/registry.json`; the weekly
  refresh challenges it. Called the incumbent before 2026-08-17.
- **Dual-view model** and **branch model.** The two architectures: the served
  CNN over a global and a local phase-folded view, and the ExoMiner-inspired
  rebuild with one convolutional branch per diagnostic view.
- **View, view set, shard set.** A view is one phase-folded or derived array a
  model consumes; the view set is the eleven-plus views of the branch model;
  the shard set is that view set written as TFRecords for training.
- **Arm.** One full cross-validated training run in a controlled comparison;
  arms come in pairs that differ in one thing (a branch dropped, a weighting
  applied). Letters name the arms of a stage; primes mark a rebuild of the same
  arm on a new shard set.
- **Control arm.** A run of the harness that scores real hosts with no injected
  transit; a model that passes such hosts is scoring the star, not the transit.
- **Noise floor.** The seed-to-seed spread of a metric measured in the same
  run, `2 x sd / sqrt(n_models_per_fold)`; a margin inside it is not a result.
- **Recall @1% FPR.** The fraction of planets caught at the threshold where one
  percent of false positives pass: what "would this reach the shortlist" means.
- **Pre-registration.** How a result will be read, written before the run
  finishes; a result outside it is reported as falsified.
- **Stages, steps, phases.** Stages 1–12 are the science stages of the record;
  steps 1–8 are the delivery plan in `PLAN.md`; Phases 0–3 are the
  pre-registered order of 2026-08-20.
- **W1–W14.** The weakness register in `known-limits.md`.
- **DV.** The SPOC pipeline's Data Validation report for a TESS target, the
  source of difference images and diagnostic scalars; `dv_usable` marks rows
  that have one. **FA** is a false alarm, a signal with no astrophysical
  source. **TCE** is a threshold-crossing event, SPOC's detection unit.
