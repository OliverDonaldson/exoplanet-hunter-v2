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

---

## Audit outcome (2026-07-13) — appended by the audit session

The document above is the historical handover; this section records how the
audit resolved each target. Fuller detail lives in the git log.

**The expansion run**: finished CV 2026-07-12 13:27 (the flow process died
right after — machine slept — so gate + publish were run manually). Run
`cebb0fe6` PROMOTED: CV ROC-AUC 0.9508 ± 0.0085 vs incumbent 0.8741, on
4,813 examples (2,448 Kepler + 2,365 TESS). It shipped with a calibration
regression (ECE 0.136 vs 0.031 — systematic under-confidence that
temperature scaling cannot correct), fixed the next day: Platt scaling in
the trainer, an ECE guard in the promotion gate, and an in-place
recalibration of the run (`pipeline/scripts/recalibrate_run.py`) — pooled
OOF ECE now **0.006**, Brier **0.087**, thresholds ~0.4.

Audit targets, item by item:

1. **Docker image** — still unbuilt/untested. Deliberately deferred with
   Fly.io (deploy phase); nothing else blocks on it.
2. **GitHub Actions CI** — observed green 2026-07-13 (10/10 runs, ~2 min
   each, including the calibration merge).
3. **Console panel parity** — still open (odd/even overlay, periodogram,
   centroid track). Deferred to the app phase by decision.
4. **Kepler subsample churn** — fixed: catalogue subsampling is now
   content-keyed (`_stable_sample`, md5 of seed:tic_id) instead of
   positional, so refresh-trigger counts reflect real pool changes only.
   NOTE: the switch causes a one-time membership change of the Kepler
   block on the next refresh; expect one legitimate retrain trigger.
5. **Machine-specific paths** — fixed: `scripts-dev/run-api.sh` discovers
   the conda env (override `$EXO_PYTHON`); v2 has its own
   `.claude/launch.json` (api + frontend).
6. **Debris** — `mlruns/0` + the file-store experiment (V1-env accident,
   35 MB) deleted; `mlruns/1` is the *live* sqlite artifact store — keep.
   `labels.previous.parquet` handling verified correct (leakage-guard
   input, versioned with the labels dir). The Kepler raw cache (30 GB)
   was MOVED from V1 into `data/raw_kepler` — v2 no longer needs
   `KEPLER_RAW_DIR` pointing across repos. V1's `data/raw` (64 GB TESS)
   was NOT deleted: v2's own TESS cache is a different, smaller set
   (17,832 vs 29,163 files), so reclaiming it is only safe if V1 never
   needs to re-run — owner's call.
7. **Dropped V1 features** — Optuna tuning and attention diagnostics
   remain unported (candidates for the next research phase; hyperparams
   date from the 881-example era).
8. **Deferred by decision** — unchanged (Fly.io, scheduled refresh,
   webhook, sky map, sequence models).
9. **Docs drift** — fixed: README build order marked complete + served
   model numbers added; OPERATING.md calibrator/gate text corrected;
   architecture.md needed no change (method-agnostic).

New findings logged during the audit — both since resolved (2026-07-14):

- `/score/{tic_id}` latency: FIXED — ephemeris resolution is user >
  catalogue > BLS (published ephemerides skip the search for known TOIs),
  and BLS itself is bounded (astropy-spaced period grid capped at 5k trial
  periods + cadence decimation). 169k-cadence target: never finished → ~25 s.
- Serving calibration mismatch: FIXED — two compounding causes measured.
  (1) Fold checkpoints on disk differed from the in-memory weights that
  scored predictions.parquet (per-example drift up to 0.31); the trainer now
  reloads the checkpoint before scoring ("score what you ship"), and
  cebb0fe6 was rescored + recalibrated from its checkpoints
  (`recalibrate_run.py --rescore`): AUC 0.9502, Brier 0.0882, ECE 0.0079.
  (2) Serving fed MC-Dropout means to calibrators fitted on deterministic
  scores — measured cost ~0.08 ECE; the calibrated headline now comes from
  the deterministic pass, MC contributes only prob_std. Residual ~1e-3
  per-request jitter (suspected single-example TF/Metal nondeterminism) is
  immaterial. The raw-score shift itself is present in-sample (mean prob
  0.41 vs 0.53 base rate on fold-0 train rows), so it is a property of the
  training objective rather than a generalization gap — Platt absorbs it.
