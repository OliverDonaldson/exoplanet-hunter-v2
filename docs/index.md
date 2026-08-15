# Exoplanet Hunter V2 — documentation

A self-refreshing, self-validating transit-detection platform: catalogue
refresh → validation gates → tf.data pipeline → calibrated 5-fold CNN ensemble
→ promotion gate → live FastAPI scoring → React vetting console.

Start with [OPERATING.md](OPERATING.md) if you want to *run* something, and
[architecture.md](architecture.md) if you want to know why it is shaped this way.

## Documents

| doc | what it covers |
|---|---|
| [architecture.md](architecture.md) | the V2 design: components, data flow, why each boundary is where it is |
| [OPERATING.md](OPERATING.md) | runbook — refresh, retrain, promote, deploy, recover |
| [DEPLOY.md](DEPLOY.md) | Docker image, Fly.io API, Render console |
| [features.md](features.md) | every model input and where it comes from |
| [data_provenance.md](data_provenance.md) | catalogue sources, label definitions, known archive quirks |
| [pipeline_diagram.md](pipeline_diagram.md) | the refresh DAG |
| [roadmap.md](roadmap.md) | the ExoMiner-inspired rebuild, staged — and the permanent old→new stage-number mapping |
| [handover-2026-08-15.md](handover-2026-08-15.md) | **start here** — audit the 2026-08-14/15 session first, then finish stage 10.5's control-arm pass |
| [handover-2026-08-14.md](handover-2026-08-14.md) | superseded by 2026-08-15. Dated, not stale — do not edit |
| ~~plan-2026-08-09.md~~ | **deleted 2026-08-14, merged into `roadmap.md`** — the ordered plan is now §4, the weakness register §1d, the standing audits §5. Handovers dated before 2026-08-14 still point at it; they are historical records and are not edited, so read §4 instead |
| [handover-stage-8-close.md](handover-stage-8-close.md) | superseded by 2026-08-14, which corrects two of its statements. Dated, not stale — do not edit |
| [handover-stage-8.md](handover-stage-8.md), [handover-2026-08-09.md](handover-2026-08-09.md), [handover-2026-08-08.md](handover-2026-08-08.md) | earlier session briefs: what is being built, what "finished" means, the non-negotiables, the trap list. Dated, not stale — do not edit |
| [audit-2026-08-07.md](audit-2026-08-07.md) | the full audit of the model and every recorded metric — read before interpreting any earlier result |
| [figures/](figures) | generated performance and provenance figures |

## Code map

Each module has its own README with the detail.

| path | what lives there | README |
|---|---|---|
| `pipeline/` | the science: ingest, preprocess, features, training, eval, validation | [pipeline/README.md](../pipeline/README.md) |
| `pipeline/src/exoplanet_hunter/` | the installable library | — |
| `pipeline/scripts/` | Hydra entry points (build, score, gate, figures) | [pipeline/scripts/README.md](../pipeline/scripts/README.md) |
| `pipeline/vendor/` | third-party source we patch and ship | [vendor/triceratops/README.md](../pipeline/vendor/triceratops/README.md) |
| `orchestration/` | the Prefect refresh DAG + schedule | [orchestration/README.md](../orchestration/README.md) |
| `api/` | FastAPI serving (`/score`, `/candidates`, `/reliability`, `/healthz`) | [api/README.md](../api/README.md) |
| `frontend/` | React vetting console | [frontend/README.md](../frontend/README.md) |
| `docker/`, `infra/` | images and deployment config | [infra/README.md](../infra/README.md) |

## The rules that do not bend

1. **Fresh data only.** Raw FITS are an evictable cache of immutable NASA
   files; every derived artefact is rebuilt by V2 code and versioned in DVC.
2. **The `/score/{tic_id}` contract is pinned.** `api/app/schemas.py` and
   `frontend/src/api/types.ts` change together or not at all.
3. **Models ship only through the promotion gate.** Beat the incumbent's CV
   ROC-AUC without degrading Brier or ECE, or it does not become the champion.
   The gate has correctly rejected two retrains; that is it working.
4. **Verify by executing.** Documentation claiming something is done is a
   hypothesis. Run the path, check the artefact.
