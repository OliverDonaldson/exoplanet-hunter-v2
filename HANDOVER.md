# Exoplanet Hunter V2 — Handover (2026-07-10)

Written at the end of the build sprint that took V2 from an empty orphan
branch to a complete, self-running system. The next session is a **fresh-eyes
audit**: tidy-up, verification of everything claimed below, and the steps
that follow the full-scale training run currently in flight.

## What this is

A self-refreshing, self-validating exoplanet transit-detection platform:
ExoFOP/NASA catalogue refresh → Pandera validation gates + leakage guard →
tf.data training pipeline → calibrated 5-fold CNN ensemble (MC-Dropout,
temperature scaling) → champion/challenger promotion gate → live FastAPI
scoring → React vetting console. Design doc: `docs/architecture.md`.
Plain-language manual: `docs/OPERATING.md`.

## Where everything lives

- **Worktree**: `/Users/ollie/Project/v2`, orphan branch `v2` — pushed as
  `main` to **github.com/OliverDonaldson/exoplanet-hunter-v2** (private,
  remote name `v2origin`). V1 lives untouched at `/Users/ollie/Project`.
- **Data/models**: DVC → Cloudflare R2 bucket `exoplanet-hunter-v2`
  (endpoint in `.dvc/config`; credentials ONLY in `.dvc/config.local`,
  which exists only on this machine — do not lose it).
- **Deploy**: Render Blueprint (`render.yaml`) serves the **static console
  only**; the API service block is commented out, deferred to Fly.io.
- **Experiments**: MLflow sqlite (`make mlflow` → :5001).

## Non-negotiable rules

1. **`conda activate exoplanet-hunter-v2` before anything.** The V1 env
   (`exoplanet-hunter`) shadows the package with V1's code — this has
   silently run the wrong trainer once already.
2. **Fresh data only.** Raw FITS under `data/raw*` are an evictable cache
   of immutable NASA files; every derived artefact must be rebuilt by V2
   code and versioned via DVC.
3. **The `/score/{tic_id}` contract is pinned**: `api/app/schemas.py` ↔
   `frontend/src/api/types.ts` change together or not at all.
4. **Models ship only through the promotion gate** (beat incumbent CV
   ROC-AUC, Brier not degraded) → `models/registry.json` (in git) names
   the served run.

## State at handover

- All 7 build-order branches merged into `v2` == GitHub `main`
  (tfdata-pipeline, validation-gates, dvc-versioning, fastapi-serving,
  dashboard, orchestrator, data-scaling) plus follow-up-metrics, the
  candidate-catalogue console, reliability diagram, and deploy prep.
- Tests: 68 pipeline + 14 api, all green; pre-commit (ruff/mypy) clean.
- **Incumbent model**: run `e5388ed9`, 5-fold CV ROC-AUC 0.8741 ± 0.034,
  Brier 0.1430, ECE 0.031 — trained on the fresh 881-example TESS-only
  build (8-dim aux). Served locally, verified end-to-end in the console.
- **IN FLIGHT**: the full-scale expansion run
  (`refresh_pipeline.py --force-train --data-config full`): 5,155 targets
  (2,655 TESS uncapped + 2,500 Kepler), 9-dim aux (centroid restored),
  ~4,200 fresh downloads → expect 24–40 h total. The flow itself runs the
  promotion gate and DVC publish at the end — no manual steps needed.

## When the expansion run finishes

1. Read the tail: CV summary + `promotion gate: PROMOTED|rejected`.
2. If PROMOTED: restart the API (`make api`) — it serves the new run;
   check `/healthz` shows the new run id; eyeball `/reliability` (ECE) and
   re-score a couple of known targets in the console (a KP and an FP).
3. If rejected: incumbent keeps serving; the run is in MLflow — compare
   fold tables before deciding anything.
4. Either way `dvc push` already ran in-flow; `git status` should be clean
   except possibly staged `.dvc` pointer bumps → commit those.
5. Worth recording: compare against V1's report numbers now that Kepler +
   centroid are back (V1's headline was on 3,275 examples, 9-dim aux).

## Audit targets (known loose ends, honestly listed)

1. **`docker/api.Dockerfile` + `api-entrypoint.sh` have never been built
   or run** — the dvc-pull-at-boot flow is designed but unverified. Test
   locally (needs Docker Desktop) before any Fly.io attempt.
2. **GitHub Actions CI has never been observed green** on the pushed repo
   — check the Actions tab; the workflow installs full TF so it may need
   caching/timeout attention.
3. **Console panel parity vs V1's six-panel figure**: odd/even *overlay
   series*, opt-in BLS periodogram, centroid *track* plot still pending
   (numbers exist in the API; plots don't).
4. **Kepler subsample churn**: `full.yaml` uncaps TESS (trigger now exact
   there) but Kepler is still sampled 1,250+1,250 → residual refresh-
   trigger noise on the Kepler side.
5. **Machine-specific paths**: `scripts-dev/run-api.sh` hardcodes the
   conda path; `.claude/launch.json` lives in the V1 repo dir.
6. **Debris**: `mlruns/` in v2 root (V1-env artefact, safe to delete);
   check `data/labels/labels.previous.parquet` handling; V1's original
   `Project/data/raw` (64 GB) is reclaimable — v2 has its own copy.
7. **Dropped V1 features not yet rebuilt**: Optuna tuning (`conf/train/
   tune.yaml` was deliberately not ported), Dash viz (superseded), and the
   attention-diagnostics module (V1 history only).
8. **Deferred by decision**: Fly.io API hosting; scheduled refresh
   (cron/GHA or `refresh_pipeline.serve(cron=...)`); the ~10-line
   new-candidate notification webhook; astropy-healpix sky map;
   sequence-model research branches (post-V2 per the design doc).
9. **Docs drift**: README build-order checklist predates completion;
   verify OPERATING.md against reality after the expansion run.

## Quick verification for fresh eyes

```bash
cd /Users/ollie/Project/v2 && conda activate exoplanet-hunter-v2
git log --oneline -15        # the story
make test                    # 82 tests green
make validate                # data gates on current artefacts
dvc status -c                # local vs R2
make api & make frontend     # then click something in the console
```

## Key documents

- `docs/architecture.md` — the original V2 design doc (committed; no need
  to re-attach it in chat).
- `docs/OPERATING.md` — plain-language runbook.
- `docs/exofop_calculations.pdf` — NExScI TSM/ESM recipes (implemented in
  `features/followup.py`, pinned to its worked example).
