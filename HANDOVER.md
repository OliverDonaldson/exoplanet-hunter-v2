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

---

## Deployment outcome (2026-07-16) — appended after the launch sprint

The system is publicly live: API at https://exoplanet-hunter-api.fly.dev
(Fly.io, syd, shared-cpu-1x/2GB, suspend-on-idle), console at
https://exoplanet-hunter-console.onrender.com (VITE_API_BASE wired).
End-to-end verified in the browser: TOI-1469.02 scored 0.99 live with
phase views and the odd/even 3.4σ caution firing.

Deploy-sprint fixes worth knowing about (details in the git log):

- `.dockerignore`'s `models/cv/*/` also swallowed the `*.dvc` pointer files
  (Docker strips the trailing slash) — the original crash-loop cause.
- pip needed `docker/constraints.txt` (training-env pins) to resolve, and
  `build-essential` in-layer for batman-package's source build.
- MC-Dropout is drawn in one batched forward pass; sequential passes ran
  >12 min on the shared CPU. API default n_mc is 20.
- Concurrent scores of one TIC could rewrite a FITS under the other's
  astropy memory-map (SIGBUS, exit 135): an existing file is now a cache
  hit regardless of the manifest, and the API serializes scoring.
- Speed package: ensemble preloads at boot, /score responses are cached
  for the process lifetime (repeat click ≈ 0.25 s), suspend-on-idle keeps
  the model in RAM across wake-ups.

---

## Re-tune + vetting-review outcome (2026-07-21) — appended after the campaign

**Served model is now run `ca906040`** (deployed 2026-07-19, verified live):
per-fold CV ROC-AUC 0.9581 ± 0.0057, Brier 0.0791, ECE 0.0276 vs incumbent
cebb0fe6 0.9502/0.0882/0.0366; pooled serving ECE 0.0129 (/reliability).
Path there: the Optuna harness shipped broken (MLflow nesting crash +
optuna never declared) — fixed b36bd38, hardened (sqlite-resumable study,
tested via `run_study`); 33 cheap trials found a flat optimum near the old
defaults (adopted: lr 3.2e-4, dropout 0.36 — 6245a75); the full 5-fold
refresh run promoted through the gate and absorbed the one-time Kepler
membership change. Verified live: CP TIC 261136679 → 0.965, FP TIC
50365310 → 0.003 with the centroid caution firing.

Ops since: weekly refresh plist is LOADED (Sat 09:00; "Load failed: 5" from
launchctl means already-loaded). Flow publish is now an allowlist
(`publishable_cv_dirs`, 261032d) after it swept 32 tuning-trial dirs to R2;
debris reclaimed with `dvc gc -c --all-commits` (297 objects; recipe in
OPERATING.md — do NOT use `-w`, it prunes committed history). Gate-rejected
runs are no longer pushed to R2 by design. 112 tests green.

**Vetting-tools review (2026-07-17, in-chat)**: compared LEO-Vetter
(Kunimoto 2025, AJ 170:280 — Robovetter-style metric/threshold vetter,
GPL-3.0, pip `leo-vetter`) and DAVE (dormant since 2021) against V2.
Verdict: complementary, not competing — they are expert test batteries, we
are a calibrated classifier. Adoption roadmap lives in the next-session
handover prompt + project memory; headline gaps: no explicit
secondary-eclipse test (only implicit in the global view), no
junk/false-alarm tests for BLS-found ephemerides (model never trained on
that regime), and a train/serve mismatch where the `snr` aux is
NaN→imputed for every non-TOI target at serve time (`_exofop_snr`).
Paper PDF: ~/Downloads/Kunimoto_2025_AJ_170_280.pdf.

---

## Vetting-cautions + model-features + perf sprint (2026-07-22)

This arc turned the LEO-Vetter review (above) into shipped code, then added
the model-level features it flagged, then absorbed a Copilot performance
review. Commits `15d19f3`..`bbdfa2a` on `v2` == GitHub `main`.

**Serving cautions — LIVE (Fly + Render), verified end-to-end.** Four
LEO-Vetter tests, each a *caution* not a gate (`prob_calibrated` stays the
headline): they add numbers + a boolean to `ScoreResponse` and a console
readout, mirroring the odd/even + centroid pattern. All new response fields
are OPTIONAL, so old clients keep working and deploy order never matters.
- Unphysical duration (§3.4, `15d19f3`): q vs q_circ from stellar density,
  a/R* from Kepler-3 (not a model fit — noted deviation).
- Odd/even timing (§4.4 Eq 13, `8a417c2`): flux-weighted per-transit
  midtimes, 10σ threshold — catches eccentric EBs at half period.
- Significant secondary (§3.9+§4.3, `9b88404`, F_red added `b4277bb`):
  simplified box-scan Model-Shift, MS4/5/6 with FA1=FA2=√2·erfcinv(Tdur/P)
  (Thompson 2018 Eq 13-14, N_TCEs=1); occultation escape hatch (depth
  ratio <10% + albedo <1); real F_red from the sig series, MS4 ignored when
  F_red>1.8. Simplifications documented in the docstring.
- FA bundle (§3.3/3.5/3.6/3.12, `4b062c7`): SWEET, asymmetry, depth
  mean/median, gap fraction — computed ONLY when ephemeris source=="bls"
  (the model never trained on junk), surfaced as one grouped low-trust chip.

**Model-level features — CODE COMPLETE, retrain IN FLIGHT.** Closes the
train/serve `snr` mismatch and feeds the diagnostics into the model:
- `features/noise.py` pink-noise transit SNR (§2.1 Eq 1-3, `fcb027f`):
  computed from the light curve so it exists for every target at train AND
  serve time — unlike the catalogue `_exofop_snr` (NaN→imputed for non-TOIs).
- 13-dim vetting-aux layout (`25a3e5c`): `[teff, radius, logg, tmag, depth,
  duration, log_period, pink_snr, centroid_snr, oe_depth_σ, oe_timing_σ,
  secondary_sig, q_ratio]`. pink_snr replaces the catalogue snr at idx 7;
  centroid stays at CENTROID_COL=8 so the fitted aux pipeline is unchanged.
  Serving `_aux_row` branches on the bundle's `aux_dim`: ≥13 builds the new
  layout, 8/9-dim legacy bundles serve BYTE-IDENTICALLY (verified: EB score
  0.0033255746024988377 unchanged after restart). Deploy-order-safe.
- **STATUS: Ollie is running the rebuild+retrain now** (`refresh_pipeline.py
  --force-train`, started 2026-07-22 ~12:40; multi-hour build+train). Serving
  is still `ca906040` (9-dim) until the new run promotes + is fly-deployed.
  An aux-only change does NOT trip the refresh trigger — this run had to be
  manual; Saturday's plist would not have retrained.

**Perf/quality (Copilot review triaged, `6b9da3e` + `bbdfa2a`).** Applied:
/score cache FIFO→LRU touch-on-hit; download_one 3s-spaced transient retry;
score_candidates.py default cv_dir now reads registry.json (was a dead V1
hash) + aborts if the bundle is 13-dim (it still builds legacy 9-dim aux —
rework before the next shortlist run); parallel download stage (manifest
threading.Lock + atomic tmp-replace write; `download_many(workers=N)` with
(mission,tid) dedup; score_candidates prefetches all targets at 4 workers
before the sequential TF loop). Rejected with receipts (in project memory):
per-TIC score lock (1-vCPU Fly box + SIGBUS history), prefetch reorder,
index copies, removing the score-what-you-ship checkpoint reload, session
pooling, threading the scoring loop.

## The pasted "Track A / Track B" list is SUPERSEDED

A prior summary listed console panels, automation, Optuna, uncertainty eval,
and since-confirmed as "remaining." All shipped before this arc:
- Console vetting panels (odd/even overlay, opt-in periodogram, centroid
  track): DONE `6220f0c`.
- Automation (weekly launchd refresh + new-candidate webhook): DONE
  `569b2fb`; plist loaded 2026-07-21 (fires Sat 09:00).
- Optuna re-tune: harness rebuilt `b36bd38`, campaign adopted `6245a75`,
  full run `ca906040` promoted + deployed 2026-07-19.
- Uncertainty validation: DONE (`uncertainty_eval.py`) — MC-Dropout std
  barely predicts errors (AUROC 0.545) vs distance-to-threshold 0.769 → NO
  abstain band; figure `docs/figures/risk_coverage.png`.
- Tidy-ups it named are closed: honest cold-start copy + DEPLOY.md
  suspend/remote-only + SIGBUS notes were `41ddc09`.

## What actually remains — do in this order

1. **Land the 13-dim retrain (IN FLIGHT — immediate).** When the flow
   finishes: read the log tail for the CV summary + `promotion gate:
   PROMOTED|rejected`; confirm the new bundle is `aux_dim==13`
   (`joblib.load(models/cv/<run>/fold_0/cnn_calibrator.joblib)["aux_dim"]`
   — the whole point). If PROMOTED: registry + `.dvc` pointers update
   in-flow → commit the bumps; Ollie runs `fly deploy --remote-only -a
   exoplanet-hunter-api`; verify live (/healthz = new run, /reliability ECE,
   EB TIC 50365310 → cautions fire, KP TIC 6892385 → clean). If rejected:
   `ca906040` (9-dim) keeps serving with no change needed (the serving
   branch handles both); compare fold tables in MLflow to see whether the
   vetting features helped or the flat optimum held.
2. **Since-confirmed holdout eval (data-gated, low effort when ready).**
   `eval_since_confirmed.py` exists (checkpointed/resumable). This retrain
   rewrites candidates.parquet → resets the holdout, so flips ≈ 0 until a
   few weekly Saturday refreshes accumulate newly-flipped dispositions. Run
   it then — it's the most convincing single prospective number. Not active
   work now.
3. **Tidy-up sweep (mostly closed — ~30 min consistency pass).** Light
   audit for drift this arc left: `score_candidates.py`'s module docstring
   still says "branch-3" / "9-dim aux" in places; a few comments predate the
   13-dim layout. Nothing functional.
4. **FINAL — UI/UX design upgrades (deliberate, scoped — memory says don't
   redesign in passing).** (a) The vetting panel grew to 6+ diagnostic rows
   this arc (centroid, odd/even depth, odd/even timing, secondary, duration,
   FA bundle) — needs hierarchy: group "cautions firing" vs "clean checks,"
   a one-line caution-summary chip row up top, consistent colour/iconography
   with the probability bar. (b) The reliability diagram Ollie finds
   confusing — rethink or replace with a plain "is it well-calibrated?"
   readout. (c) Cold-start expectation-setting, empty/error states, mobile.
