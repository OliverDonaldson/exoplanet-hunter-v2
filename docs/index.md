# Exoplanet Hunter — Documentation

A self-refreshing, self-validating transit-vetting platform: catalogue refresh →
validation gates → tf.data pipeline → calibrated cross-validated CNN ensemble →
promotion gate → live FastAPI scoring → React vetting console.

## Documents

| Document | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | environment, everyday commands, the refresh loop, scoring a target, deploying |
| [model_specs.md](model_specs.md) | every model input, where it comes from, and the two architectures |
| [model_pipeline.md](model_pipeline.md) | the end-to-end path, stage by stage |
| [roadmap.md](roadmap.md) | the record of what was measured and the plan for what remains |
| [overview.md](overview.md) | what the pipeline outputs, column by column, and its known limitations |
| [troubleshooting.md](troubleshooting.md) | failure modes, operational traps, what is safe to delete |
| [data_provenance.md](data_provenance.md) | catalogue sources, sky coverage, and the measured-findings ledger |
| [references.bib](references.bib) | bibliography |
| [figures/](figures) | generated performance and provenance figures |

Start with **getting-started** to run something, **overview** to understand what
comes out, and **roadmap** for where the project stands.

## Code map

Each module carries its own README with the detail.

| Path | What lives there |
|---|---|
| `pipeline/` | the science: ingest, preprocess, features, training, eval, validation |
| `pipeline/src/exoplanet_hunter/` | the installable library |
| `pipeline/scripts/` | entry points: build, score, gate, figures |
| `pipeline/vendor/` | third-party source patched and shipped |
| `orchestration/` | the Prefect refresh DAG and its schedule |
| `api/` | FastAPI serving |
| `frontend/` | React vetting console |
| `docker/`, `infra/` | images and deployment config |

## The rules that do not bend

1. **Fresh data only.** Raw FITS are an evictable cache of immutable archive
   files; every derived artefact is rebuilt by current code and versioned in DVC.
2. **The `/score/{tic_id}` contract is pinned.** `api/app/schemas.py` and
   `frontend/src/api/types.ts` change together or not at all.
3. **Models ship only through the promotion gate.** Beat the champion's
   cross-validated ROC-AUC without degrading Brier or ECE, or it does not become
   the champion. The gate has correctly rejected several retrains; that is it
   working.
4. **A margin smaller than its noise floor is not a result.** Every run measures
   and reports its own floor. The rule is `2 x sd / sqrt(n_models_per_fold)`.
5. **Pre-registration is binding.** A result landing outside the terms fixed
   before it was run is reported as falsified, never re-specified.
6. **Verify by executing.** Documentation claiming something works is a
   hypothesis. Run the path, check the artefact.
