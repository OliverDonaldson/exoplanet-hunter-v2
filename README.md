# Exoplanet Hunter V2

Self-refreshing, self-validating, self-serving transit-detection platform:
catalogue refresh → validation gates → tf.data training on an on-demand GPU
burst → calibrated 5-fold ensemble → live FastAPI scoring → interactive
React vetting console. Governing principle: **beat the baseline before you
cheer** — every component must beat the simplest thing that already works,
or it doesn't ship.

Architecture reference: `docs/architecture.md` (the V2 design document).

## Layout

```
pipeline/    ML pipeline package (exoplanet_hunter) + Hydra conf + scripts
api/         FastAPI serving layer — /score/{tic_id}, pinned contract in app/schemas.py
frontend/    React vetting console (Vite + TypeScript)
docker/      api / frontend / GPU-burst-train images  (compose file at root)
orchestration/  Prefect|Dagster DAG            (lands in feat/orchestrator)
infra/       R2 layout, secrets policy, hosting notes
data/        fresh artefacts only — regenerated, DVC-tracked, never committed
```

## Provenance

Seeded by clean-slate extraction from V1 (`main` @ a5faabc plus working-tree
improvements) — the battle-tested science core only:

| Salvaged | Rewritten in V2 (not ported) |
|---|---|
| preprocess: clean / flatten / fold / views | trainer + in-RAM data module → `feat/tfdata-pipeline` |
| models: dual-view CNN (SE + MHA), focal loss, MC-Dropout, RF baseline | Optuna tuning, MLflow utils (rebuilt against tf.data) |
| features: centroid (BEB vetting), handcrafted (RF) | Dash/`viz` dashboard → Streamlit + React console |
| search: BLS / TLS | attention diagnostics (V1 report artefact) |
| training/calibration: temperature scaling | registries/paths tied to V1 disk layout |
| eval: metrics, six-panel vetting figure | all preprocessed data artefacts (fresh data only) |
| data: catalogue TAP builder, downloader, stellar params | |
| scripts: build_dataset, preprocess_only, score_target/candidates, render_vetting | |

**No data artefacts were ported.** The first V2 milestone regenerates the
catalogue and views from NASA sources so the new pipeline is validated
end-to-end on data it produced itself.

## Build order (each branch leaves `v2` working)

1. `feat/tfdata-pipeline` — tf.data (map→cache→shuffle→batch→prefetch),
   TFRecord shards, mixed precision; rewrite trainer on top.
2. `feat/validation-gates` — Pandera catalogue checks in CI + the
   beats-current-best promotion gate + leakage guard.
3. `feat/dvc-versioning` — catalogue + views under DVC, R2 remote.
4. `feat/fastapi-serving` — refactor `scripts/score_target.py` into the
   `/score/{tic_id}` service; deploy container.
5. `feat/dashboard` — Streamlit explorer first (reuses matplotlib six-panel),
   then the React console against the pinned contract; reliability diagram +
   sky map.
6. `feat/orchestrator` — Prefect/Dagster DAG with conditional GPU burst.
7. `feat/data-scaling` — dataset expansion, now safe because 1–6 made it
   automated and validated.

## Quickstart

```bash
conda env create -f environment.yml && conda activate exoplanet-hunter-v2
make test        # salvage smoke tests + API contract tests
python pipeline/scripts/ingest_exofop.py   # build the candidate catalogue
make api         # FastAPI on :8000 (docs at /docs)
make frontend    # console on :5173 (needs: cd frontend && npm install)
```

The console's catalogue page (and `GET /candidates` + `/candidates.csv`)
serves the normalised ExoFOP TOI+CTOI table — see `data/README.md` for the
source exports and rebuild command.
