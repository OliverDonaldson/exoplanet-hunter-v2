# Exoplanet Hunter V2 — Handover (2026-07-10)

> **2026-08-07 — read `docs/audit-2026-08-07.md` before trusting any number
> below.** A full audit of the model and every recorded metric found the figures
> arithmetically correct but the comparisons behind them invalid: the incumbent
> baseline predated the entire Stage 1 rebuild, the TESS promotion gate never
> engaged, K2 was in no comparison, and four separate defects meant neither
> stage 2(a) run was a fair test of the architecture. All fixed. The stage 2(a)
> readings in this file are superseded by the roadmap's audit section.

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
2. **Archive-expansion data pass (queued after Step 1 — the next big
   data/model pass).** Broaden the label base beyond TESS-TOI + Kepler-KOI and
   add the NASA Exoplanet Archive's predicted observables (see
   `docs/data_provenance.md`). Do the sub-steps in this order — risk rises down
   the list, and (a) ships on its own without a retrain:

   **2a. POE observables — self-contained, ships without a retrain.**
   - In `features/followup.py` implement, from the NASA POE equations:
     stellar luminosity `L* = 4π R*² σ Teff⁴` (when not given), **insolation
     flux** `S = L*/d²` in Earth units (inverse-square), and **habitable-zone
     radii** (Kasting recent-Venus / early-Mars, 0.75 & 1.77 AU for the Sun,
     scaled by √L*).
   - Add `insolation_earth`, `hz_inner_au`, `hz_outer_au` columns to the
     candidate catalogue → `api/app/schemas.py::CandidateRow` **and**
     `frontend/src/api/types.ts` (pinned contract — move together) → render in
     the console.
   - Cross-check the existing transit-depth + RV forms against POE's.
   - Unit-test to the worked case: Earth→Sun gives S≈1 and HZ 0.75–1.77 AU.
   - Ship as its own commit; no model change.

   **2b. Cleaner Kepler negatives — TAP-only, feeds the next build.**
   - In `data/catalog.py` add `_query_certified_fp()` against the **`fpwg`**
     (Certified False Positives) table, and optionally `koifpp` (FP
     probabilities).
   - Use it to confirm/upgrade the `koi_disposition == 'FALSE POSITIVE'`
     negatives (higher-quality negative labels). No new download path.

   **2c. K2 mission integration — the big one.**
   - `_query_k2()` in `data/catalog.py` against **`k2pandc`**: map disposition
     → label, normalise period/t0/depth/duration units, `mission="K2"`, key on
     the **EPIC** id.
   - Add a K2 fetch path to `data/download.py` (`lightkurve` `mission="K2"`,
     campaign-aware; K2 light curves are on MAST — a third branch beside TESS
     SPOC and the Kepler direct-archive path).
   - EPIC stellar params in `data/stellar.py`.
   - Wire `n_confirmed_k2` / `n_false_pos_k2` into `CatalogRequest` +
     `build_label_catalog`. K2 adds ecliptic-plane coverage — a third band on
     the sky map.

   **2d. Rebuild + retrain + promote + deploy.**
   - Run `refresh_pipeline.py --force-train` with the expanded config; the gate
     must beat the incumbent (AUC up, Brier/ECE not degraded).
   - Commit registry + `.dvc` pointer bumps; Ollie `fly deploy`s; regenerate
     the `docs/data_provenance.md` figures (K2 band now visible); verify live.

3. **Since-confirmed holdout eval (data-gated, low effort when ready).**
   `eval_since_confirmed.py` exists (checkpointed/resumable). Retrains
   rewrite candidates.parquet → reset the holdout, so flips ≈ 0 until a
   few weekly Saturday refreshes accumulate newly-flipped dispositions. Run
   it then — it's the most convincing single prospective number. Not active
   work now.
4. **Tidy-up sweep (mostly closed — ~30 min consistency pass).** Light
   audit for drift this arc left: `score_candidates.py`'s module docstring
   still says "branch-3" / "9-dim aux" in places; a few comments predate the
   13-dim layout. Nothing functional.
5. **FINAL — UI/UX design upgrades (deliberate, scoped — memory says don't
   redesign in passing).** (a) The vetting panel grew to 6+ diagnostic rows
   this arc (centroid, odd/even depth, odd/even timing, secondary, duration,
   FA bundle) — needs hierarchy: group "cautions firing" vs "clean checks,"
   a one-line caution-summary chip row up top, consistent colour/iconography
   with the probability bar. (b) The reliability diagram Ollie finds
   confusing — rethink or replace with a plain "is it well-calibrated?"
   readout. (c) Cold-start expectation-setting, empty/error states, mobile.

## Step-2 data-expansion + source-review sprint (2026-07-24)

Ran the whole archive-expansion data pass (2a–2d), turned a user-supplied source
review into shipped vetting code, and built a statistical-validation layer.
Commits `8a8a2d5`..`efb4d73` on `v2` == GitHub `main`. **Serving is UNCHANGED —
`ca906040` (9-dim) still live on Fly; nothing deployed.**

**STEP 2d RETRAIN — REJECTED by the gate (run `856872ad`, 13-dim aux + K2).**
The payoff run of the whole data pass did NOT beat the incumbent: CV ROC-AUC
**0.95768** ± 0.0072 vs ca906040 **0.95813** ± 0.0057 (Δ −0.00046, lost on the
primary metric), Brier 0.07881 (a hair better), ECE 0.02818 (a hair worse), on
**5,380 examples** (K2 landed: +562 over the 4,818 TESS+Kepler set). Registry
unchanged; `856872ad` has no `.dvc` (rejected runs aren't published). Same null
as the 2026-07-23 13-dim rejection: the expanded data + vetting aux yield a model
statistically indistinguishable from (marginally under) the incumbent. **K2 did
not move the CV headline** — plausibly K2 is harder (patchy stellar params,
reduced-pointing systematics) and/or the model is at a task ceiling; in-
distribution CV also can't reward K2's cross-mission generalization. The gate
worked as designed.

**Everything shipped (all in the build; none beat the incumbent on CV, but all
correct + serving-safe):**
- **2a POE observables (`ce5b669`)** — insolation + habitable-zone in
  features/followup.py through the pinned contract + console columns; literature-
  validated (TOI-715 b S=1.53 vs pub 1.56). *I regenerated candidates.parquet
  locally but did NOT dvc-push — the live console won't show the columns until
  `ingest_exofop` + `dvc push` (or the weekly refresh).*
- **2b cleaner Kepler negatives (`0b3ea30`)** — fpwg/koifpp are RETIRED from the
  archive (TAP + legacy both gone); reconstructed via DR25 koi_score
  (`_query_certified_fp`: FALSE-POSITIVE with koi_score < 0.5, ~79% certify).
- **2c K2 integration (`7ed5603`)** — `_query_k2` against k2pandc, EPIC-keyed,
  mission="K2". GOTCHA: default_flag=1 omits the ephemeris for RV-confirmed
  planets → require period+epoch+duration, prefer default → 1,364 EPIC stars
  (315 CP / 215 FP / 834 cand). Depth percent (÷100, verified). download.py K2
  cfg (lightkurve path); build_dataset/preprocess_only group K2 with Kepler
  (EPIC ≠ TIC — never the TESS stellar fetch). full.yaml K2 uncapped.
- **Log-transform aux (`8a8a2d5`)** — signed-log pink_snr + secondary_sig before
  StandardScaler (linear probe +0.036 AUC on pink_snr; MLP within noise, so it
  didn't flip the gate). `_log1p_centroid` untouched (live model pickles it).
- **Source review** (Kepler DV Twicken 2018 + Vetting-Detection-Efficiency
  Coughlin 2015 + SDET Morgan 2019 + TRICERATOPS Giacalone 2021 — PDFs in
  ~/Downloads): our methods match the references. Shipped: **T_p secondary
  thermal arm (`04443b0`)** (DV Eq 3 — reflected OR thermal excuses a shallow
  secondary, rescues hot Jupiters); **TRICERATOPS FPP/NFPP validation layer
  (`3f640da` +fixes)** — optional offline shortlist-time layer, `pip install -e
  'pipeline[validation]'`. **NFPP VERIFIED** against the paper (WASP-156b NFPP
  0.00 exact — the nearby-EB discrimination our CNN lacks). Hard-won: pytransit
  import shims (`485623c`), TRILEGAL SSL bypass (`8009ef8`, its cert is
  unverifiable), SPOC-aperture vs 5x5 (`c6f8c81`). **FPP verdict: stop chasing
  exact reproduction** — for bright/isolated targets the aperture barely moves it
  and the WASP-156b gap (0.75 vs 0.33) is raw-SAP LC prep, not the aperture; use
  NFPP as the headline, FPP as directional. **Injection-recovery core (`efb4d73`)**
  — eval/injection_recovery.py tested core (inject_box_transit / transit_snr /
  completeness_curve); runner deferred to the served model.

## Final steps — the path to done (in order)

1. ~~**Resolve the 2d rejection + reconcile data.**~~ **DONE 2026-07-25** — see
   the section below.
2. ~~**Propagate 2a to production.**~~ **DONE 2026-07-25** — see below.
3. **Build the validation runners against the served model.**
   (a) ~~injection-recovery runner~~ **DONE 2026-07-25** — 50% completeness at
   S/N ≈ 15, 90% at ≈ 44 (baseline-corrected); see the section below.
   (b) **NEXT — the FPP/NFPP shortlist run.** `score_candidates.py` first
   (~4,685 TESS candidates, ~45 min, resumable, checkpoints every 25 rows),
   then `validate_candidates.py --insecure-trilegal` — minutes per target, so
   terminal-first. Commands in "Step 3(b) — how to run it" below. Optional
   review gaps still open here: ephemeris-match test, statistical-bootstrap FA
   (DV §3.5).
4. **Step 3 — since-confirmed holdout eval.** Data-gated: needs a few weekly
   Saturday refreshes to accumulate flips, then run eval_since_confirmed.py.
   **The clock starts 2026-07-25** — that is the first Saturday the cron
   actually published, so earlier runs accumulated nothing.
5. **Step 4 — tidy sweep.** The aux unification landed (`78b37a2`) and
   `score_candidates.py` is rebuilt on it. What remains is the numbered list in
   "Still open" at the end of this file — `preprocess_only.py` first, it can
   still silently destroy the data-of-record — plus stale docstrings.
6. **FINAL — UI/UX redesign (the locked last step; do NOT do it in passing).**
   Mission Control aesthetic, **manus** north star (source ZIP `~/Downloads/
   Exoplanet Hunter UI Webpage Design for Vetting Console.zip`; Tailwind v4 +
   shadcn + recharts; teal #4DFFD2 / amber #F5A623 / bg #050608; Space Grotesk/
   Inter/JetBrains Mono). Landing mockup:
   https://claude.ai/code/artifact/c81f29d4-a6cb-46b8-bb1d-c2a038991701. Decisions:
   adopt Tailwind+shadcn (lift manus near-directly), three.js/R3F hero-only, no
   anime.js. Open feedback: redesign the radar bootloader ("scanning a planet",
   full sweep), study manus transitions, drop the two lens-flare glints, make
   Upload first-class (small backend: FITS→preprocess→score; RA/Dec→MAST cone-
   search→TIC→score).

## Rejection resolved + weekly-refresh data bug (2026-07-25)

Commits `0898939`..`f640318`. **Serving still `ca906040` (9-dim) on Fly —
untouched, nothing deployed.** 173 tests green, tree clean, R2 in sync.

**Verdict on the 2d rejection: KEEP, and the null is real.** The fold-table
Δ (−0.00046 AUC) is **0.10 standard errors** — Welch p=0.92 across all five
metrics. Better than the fold table, a *paired* comparison on the 4,610
targets both runs scored out-of-fold:

| benchmark | 856872ad (13-dim +K2) | ca906040 (9-dim) | Δ |
|---|---|---|---|
| pooled OOF AUC, **raw** scores | 0.95208 | 0.95195 | **+0.00014** (95% CI −0.0038…+0.0043) |
| — TESS rows only (n=2,372) | 0.90566 | 0.90439 | +0.00127 |
| — Kepler rows only (n=2,238) | 0.98861 | 0.98943 | −0.00082 |

The two models are **indistinguishable in discrimination**. Method note worth
keeping: on *calibrated* probabilities the same paired test reads −0.00371
with a CI excluding zero — that is an **artifact of pooling OOF probabilities
across folds with different Platt (a,b)**, not a real regression. Compare raw
scores when comparing runs; anything that pools calibrated OOF across folds
inherits this.

**K2 was not the problem — the handover's "K2 is harder" guess was wrong.**
K2 is the *easiest* slice: AUC **0.9606** and Brier **0.0646** on its 527 rows,
vs 0.9570 / 0.0804 on TESS+Kepler. Including K2 *raised* the headline slightly
(all-rows 0.95768 vs TESS+Kepler-only 0.95700). Where the real headroom is:
**TESS AUC 0.906 vs Kepler 0.989** — an 8-point gap, on the mission we actually
serve. That, not more archives, is the lead for any future model work.

**Found while reconciling: the weekly refresh was silently destroying the
data-of-record.** Today's 09:00 Saturday cron rewrote `data/labels/` from the
5,686-row K2 build to a **1,000-row TESS-only** catalogue.

- Cause A — `refresh_label_catalogue()` hand-rolled
  `CatalogRequest(500, 500, seed=42)` and ignored `data_config` completely, so
  `--data-config full` in the plist was decorative. Only
  `preprocess_and_shard` ever honoured it, and that runs only on a retrain.
  Fixed: stage 1 is now one implementation (`build_labels_from_cfg`) behind
  `pipeline/scripts/refresh_labels.py`, called with the same data group; the
  flow default follows argparse to `"full"`. Regression-tested against both
  config groups.
- Cause B — `publish()` shelled out to a bare `"dvc"`, which is not on
  launchd's PATH: every cron run has died there with `FileNotFoundError` and
  **has never versioned or pushed**. Fixed (resolve beside `sys.executable`).
  This failure is the only reason the clobber never reached R2 — the tiny
  catalogue was never `dvc add`ed, so the staged pointers still described the
  K2 build and it restored cleanly from cache.
- The gates did not catch a catalogue shrinking 82%; `decide_training` saw
  "0 new" and skipped. **A shrink guard in validate_data.py is worth adding.**

**Reconciled + 2a propagated.** Data-of-record is now the K2 build (labels
5,686 = 2,656 TESS / 2,500 Kepler / 530 K2; views + tfrecords 5,380 examples)
with today's fresher ExoFOP pull, pushed to R2 and pointers committed. The
candidate catalogue carries the 2a POE columns (`insolation_earth`,
`hz_inner_au`, `hz_outer_au`; 11,224 rows) — today's cron had already run
`ingest_exofop` with the POE code before dying at publish, so only the push
was outstanding. Provenance figures regenerated with the K2 ecliptic band, now
from a committed generator (`pipeline/scripts/plot_provenance.py`) instead of
ad-hoc code.

**Not done here:** the commits are local — `git push v2origin v2:main` is
outstanding. Serving needs no change.

## Step 3(a) — injection-recovery runner (2026-07-25)

Commit `d08f574`. `pipeline/scripts/injection_recovery.py` — the completeness
measurement CV ROC-AUC cannot give.

**Design decision worth keeping:** the runner does *not* rebuild the pipeline.
`TargetScorer.score()` takes an optional `inject=InjectionSpec(...)` that
multiplies a box transit into the raw flux right after `lk.read`, before
cleaning — so preprocess, the aux layout (whatever `aux_dim` the registry
serves) and the ensemble are the shipped ones. Any parallel implementation
would drift exactly the way `score_candidates.py` did.

**Each injection targets a transit S/N; the depth is solved per host**
(`depth_for_snr`). A fixed depth grid is useless here — measured host CDPP
spans 22–175 ppm across the cache, so a 500 ppm transit is S/N 99 on a quiet,
heavily-observed target and S/N 6 on a noisy one, and every injection piles
into one or two bins. Targeting S/N samples the curve where it turns over.

New in the core: `noise_ppm` (CDPP on the transit timescale — bin the
flattened flux to the duration, MAD-sigma of the bin means; computed here
rather than via lightkurve so the units are unambiguous) and `count_transits`.

Smoke run (3 hosts × P=5 d × S/N 3/7/15/50, run ca906040, threshold 0.486):
0/3 recovered at S/N 3, 3/3 at S/N ≥ 7 — a turnover between 3 and 7, on far
too few hosts to quote. The real run is the command below; it is resumable
(checkpoint parquet per injection, keyed on tic_id+period+snr_target).

```bash
caffeinate -dis /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/injection_recovery.py --hosts 40 \
  2>&1 | tee outputs/injection-recovery.log
```

~960 injections, expect ~30 min. Done when it logs "completeness by S/N bin"
and writes `docs/figures/completeness.png` + `results/injection_recovery.parquet`.

**Still open in step 3:** (b) the FPP/NFPP shortlist run
(`validate_candidates.py --insecure-trilegal`). Optional review gaps unchanged
(ephemeris-match test, statistical-bootstrap FA).

## Validation gates repaired (2026-07-25)

Commits `c5cdc37`..`bea9818`. Two follow-ups from the refresh-bug arc, both
found by pointing the gates at the restored data-of-record.

**The label-catalogue gate had been red since the K2 build (7ed5603).** The
mission domain in `validation/schemas.py` was still `["TESS", "Kepler"]`, so
all 530 K2 rows failed `validate_data.py --strict` — the exact invocation the
refresh DAG runs. It went unnoticed because the only green run since was
today's cron, which had *already* clobbered the catalogue to TESS-only rows:
the data regression masked the schema failure. Restoring the K2 build is what
exposed it. Also admitted `REFUTED` to the disposition domain — the ingest can
emit it today via `K2_DISPOSITION_LABELS`, so the next refresh pulling one
would have failed for the same reason.

**k2pandc publishes `0` as its "unknown" placeholder**, and the ADQL
`is not null` filter passes it straight through. Two live rows (EPIC
202059377, 203485624) reached labels.parquet as zero-length transits. Fixed in
the ingest rather than by relaxing the check — a literal zero-length transit
should stay a hard failure. `period` now carries the same guard (same filter,
same `gt(0)` check); **`t0` deliberately does not** — an epoch has no
positivity constraint and live K2 rows reach −1614 BTJD, so a 0 there is a
date, not a placeholder. `depth` needs no guard: it is `ge(0)`, and k2pandc
uses null for unknown depth (159 K2 rows).

**Shrink guard** (`validation/shrink.py`, gate `label-shrink`): fails when the
catalogue loses more than `--max-shrink-frac` (default 10%) of its rows, or
when any mission present in the previous catalogue drops to zero — the mission
check unconditional, since losing one of three missions can hide inside the
fraction. `--allow-shrink` overrides and logs what it allowed (Step 2b's DR25
certification legitimately retired ~21% of bare Kepler FPs).

Data-of-record repaired in place to match the ingest fix and pushed to R2;
counts unchanged (5,686 = 2,656/2,500/530, 2,901 pos / 2,785 neg). All five
gates PASS on the live artefacts, 195 tests green, tree clean.

### Injection-recovery result (2026-07-25, run ca906040, 990 injections / 40 hosts)

| S/N | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| raw | 0.26 | 0.29 | 0.34 | 0.48 | 0.63 | 0.74 | 0.86 | 0.95 |
| baseline-corrected | 0.00 | 0.04 | 0.10 | 0.30 | 0.49 | 0.64 | 0.81 | 0.94 |

**Headline: 50% completeness at S/N ≈ 15, 90% at S/N ≈ 44** (corrected). The raw
curve says S/N 10.6 — quote the corrected one.

**The raw curve is contaminated and the control arm proves it.** With *no*
injection at all, 26.4% of host×period cells still pass threshold: 46.7% for
planet hosts vs 12.3% for false-positive hosts. Hosts come from the labelled
catalogue, so half carry a real transit; folded at the wrong period it still
leaves transit-like structure, and the stellar aux describes the host either
way. The model is partly scoring the host, not the injection.

The control rate (29/110) came out bit-identical to the S/N=3 rate (29/110).
Checked, and it is coincidence, not a wiring bug: all 110 paired probabilities
differ (mean |Δ| 0.112) and 16 verdicts flip between the two arms — they
happen to flip symmetrically. Curiosity worth noting: at S/N 3 the mean Δprob
is **negative** (−0.031) — a transit too weak to detect makes the model
slightly *less* confident, presumably by perturbing pink_snr / odd-even
without adding convincing transit shape.

Completeness is genuinely period-dependent, so it is not one curve: at S/N 10,
P=3 d recovers 58% against P=7 d's 37% (more transits, better-sampled fold).

Reproduce (resumable; the control arm alone is ~120 scores on an existing run):

```bash
caffeinate -dis /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/injection_recovery.py --hosts 40 \
  2>&1 | tee -a outputs/injection-recovery.log
```

`--host-label 0` restricts the pool to false-positive hosts (12.3% baseline
instead of 26.4%) if you want a cleaner run rather than a corrected one.

**Env note:** the run was launched from a shell with the *V1* env
(`exoplanet-hunter`) active. It was still correct — the absolute interpreter
path bypasses activation entirely, verified: `exoplanet_hunter` resolved to
`/Users/ollie/Project/v2/pipeline/src/`, PYTHONPATH unset. This is exactly why
the commands use the full path rather than relying on `conda activate`.

## Audit sessions reviewed and committed (2026-07-25, later)

Commits `1431660`..`c41fa62`. Two audit sessions (security, V1-remnants) had
left work uncommitted with their findings unwritten. Every claim was
re-derived before committing; the notes below are what survived checking, not
what was reported. **Serving still `ca906040` (9-dim) on Fly — untouched.**
215 tests green, tree clean.

**Credentials never leaked, and now cannot.** `.dvc/config.local` holds the
live R2 keys. It has never been committed — no blob of that name in any of the
137 commits across all refs, and the key id and secret appear in zero commits.
The Dockerfile does a targeted `COPY .dvc/config`, so it never reached an image
layer either. The real exposure was narrower and real: `fly deploy` tars the
whole build context to the remote builder, and `.dockerignore` excluded
`.dvc/cache/` and `.dvc/tmp/` but not `config.local`. Now excluded.

**The `inf` claim is true, and worse than "rejects bad input".** Measured
against the pre-fix signature: `inf`, `Infinity` and `1e400` all satisfy a bare
`gt=0` and reach the handler (`nan` was already rejected — `nan > 0` is false).
An inf period is not a crash, it is a silent one: `((t - t0 + P/2) % P) / P`
leaves *every* cadence non-finite, so the phase fold has zero usable points.
`tic_id` was likewise unbounded — `/score/-1` and a 20-digit id both reached
MAST and the manifest.

**But the bounds as first written would have broken live traffic.** Neither
audit checked them against the catalogue. Period (max 8900 d) and duration (max
378 h) sit comfortably inside, but `t0` at ±1e6 would have 422'd three rows the
console can send today: three CTOIs carry malformed epochs — two at BJD 2.46e7,
one dated 1867 — which the console's own BJD→BTJD conversion
(`VettingPanel.tsx:217`) forwards as ~2.2e7 BTJD. They score today. Raised to
1e8 (`8bd4118`), still rejecting `inf` and `1e400`, with all three pinned as
tests. **The bounds are there to reject non-finite input, not to filter data
quality** — the moment they encode a plausible range they start rejecting real
rows.

**The cache race is real but needs help to show.** A plain contended loop finds
nothing, because CPython's 5 ms switch interval lets a short check-then-act
finish before preemption. At `sys.setswitchinterval(1e-6)` and 320k iterations
the pre-fix idiom gives **2,276 KeyErrors and 1,605 RuntimeErrors, and breaches
the 128 cap (134 entries)**; the locked version, zero and 128. Real requests
interleave on I/O, so production has the preemption points this had to force.
Worth knowing for the next time a lock looks speculative.

**Found in review, not by either audit: the 404 leaked paths too.** Both audits
redacted the 503 bodies and stopped. But `score.py` did `detail=str(exc)` on
`NoLightCurveError`, whose message interpolates `DownloadResult.reason`, and
several reasons in `download.py` interpolate the underlying exception —
`download error: {exc}`, `fits write error: {exc}`. An `OSError` carries the
absolute path it failed on. Reproduced end to end: an ENOSPC returns
`/srv/data/raw/.lightkurve/...` in the response body. Redacted at the boundary,
keeping the diagnosis and any MAST URL.

**`render_vetting.py` is alive; the loose end was a wrong column, not a dead
script.** Its input is `results/candidates_scored.parquet` from
`score_candidates.py` — exactly what step 3(b) produces. What was dead is the
branch preferring `results/discovery_shortlist.parquet`, from V1's
`discovery_shortlist.py`: that script lives only on V1's `main` and was never
ported (`git cat-file -e v2:scripts/discovery_shortlist.py` exits 128, `main`
exits 0), so nothing can write that artefact. The enrichment it carried was
*wrong*, not merely unreachable: the figure read `row["TFOPWG Disposition"]`,
ExoFOP's raw header, but the ingest renames it to `disposition`. On the live
catalogue the old key returns None on all 7,149 rows and the new one returns PC
(4,685) / CANDIDATE (2,464) — **every vetting figure v2 has ever rendered had a
blank TFOPWG subtitle.**

**Test hygiene worth keeping:** the API tests share a process-lifetime response
cache, so a TIC scored by one test was served from cache to the next and never
reached its stub. The autouse fixture now clears it. This is what made two new
404 tests pass alone and fail in suite.

### Still open — reported by the audits, deliberately not applied

Verified as still true after `78b37a2`. In rough priority:

1. **`preprocess_only.py` can silently destroy the data-of-record** — the same
   class as this morning's refresh clobber, still live. It writes
   `LEGACY_AUX_DIM` (9) into `data/processed/views.npz` where
   `build_dataset.py` writes `TRAINING_AUX_DIM` (13), and it is now the *only*
   script that bypasses `build_labels_from_cfg`, hand-rolling a
   `CatalogRequest` with no `n_confirmed_k2`/`n_false_pos_k2` — so it drops all
   530 K2 rows. `shard_views.py` then re-shards from `aux_features.shape[1]`
   and the trainer trains happily. No error anywhere. It is a *documented
   recovery script*, which is the dangerous part.
2. **`force_download=true` is an unauthenticated remote disk-fill primitive** —
   bypasses both the response cache and the FITS cache, ~116 MB staged per
   request on a 44-sector target, staging never cleaned, no `[mounts]` in
   `fly.toml` so it lands on ephemeral rootfs that `suspend` preserves. The
   console never sends the parameter. Cleanest fix is to drop it from the HTTP
   surface and keep it CLI-only.
3. **9-dim promoted model vs 13-dim shards** crashes four scripts with a raw TF
   trace (`make_performance_figures.py`, `uncertainty_eval.py`,
   `export_predictions.py`, `recalibrate_run.py --rescore` — the documented
   calibration-recovery path). Nothing guards the mismatch.
4. **`score_target.py` is the last hand-rolled aux implementation** — still
   capped at 9 dims (`bundle.get("aux_dim", 8)`, `if aux_dim >= 9`), never
   imports `build_aux_row`. Breaks the moment a 13-dim model is promoted.
5. Smaller: one `/score` request can hold the only scoring slot for minutes
   with no timeout (`force_bls` + `include_periodogram` runs two full BLS
   passes under `_score_lock`, `hard_limit = 25`); `/candidates.csv` has no row
   cap (3.5 MB/request, no cache); `/healthz` trusts `registry.json` alone and
   can report healthy on a machine whose artefacts never pulled; `make
   data-push`/`data-pull` still shell out to a bare `dvc`; `promotion_gate.py
   --models-dir` is half-honoured (`promotion.py:107` opens `cv_summary`
   relative to cwd); the flow's `train()` runs without a data group so MLflow
   names a `full` build `cnn-cv-default`; `ci.yml` lacks the catalogue/promotion
   gate jobs its own trailing comment promises, so "gates are the same code CI
   runs" is false — CI runs ruff + pytest only.

Housekeeping the security audit raised, none of it code: `chmod 600
.dvc/config.local` (currently 0644); `DVC_NO_ANALYTICS=1` in the Dockerfile
(the container phones home per cold start); check Cloudflare R2 → bucket →
Settings that the Public Development URL reads "Not enabled" (the S3 endpoint
refuses anonymous access, but an `r2.dev` binding is invisible from outside);
add "fetch all branches first" to the `dvc gc -c --all-commits` recipe in
`docs/OPERATING.md`, since it reads local git history only. Disclosure: that
audit made three live verification requests to the production API, which added
three junk entries to the download manifest on the Fly machine's ephemeral
rootfs; they clear on the next full restart.

## Step 3(b) — how to run it

Nothing has been run yet: there is no `results/candidates_scored.parquet` and
no `outputs/score-candidates.log`. Two stages, both terminal-first (the second
is minutes per target).

Stage 1 — score the TESS candidates. Hydra-style `key=value`, **not** `--out`.
~4,685 rows, ~45 min, resumable, checkpoints every 25 rows:

```bash
caffeinate -dis /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/score_candidates.py limit_mission=TESS \
  2>&1 | tee outputs/score-candidates.log
```

Stage 2 — FPP/NFPP on the top of that shortlist. argparse flags here:

```bash
caffeinate -dis /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/validate_candidates.py \
  --candidates data/labels/candidates.parquet \
  --shortlist results/candidates_scored.parquet \
  --top 20 --out results/candidates_validated.csv --insecure-trilegal \
  2>&1 | tee outputs/validate-candidates.log
```

`--insecure-trilegal` is needed because stev.oapd.inaf.it ships a broken cert
chain; only RA/Dec is sent. **Read NFPP as the headline and FPP as
directional** — NFPP reproduced the paper exactly on WASP-156b (0.00), FPP did
not (0.75 vs 0.33), and that gap is raw-SAP light-curve prep, not a bug.

Then `render_vetting.py` draws the six-panel figures from the same scored
parquet — it now reads the ingest's `disposition`, so the TFOPWG subtitle
finally populates.

## Step 3(b) blocked twice, then unblocked (2026-07-26)

Commit `b56c56f`. `score_candidates.py` aborted at the first line of the
scoring loop on two consecutive attempts — 7.4 h of downloads on the first,
31 min on the second — with `ValueError: I/O operation on closed file` from
tqdm's `status_printer`.

**It was never sleep or a dead terminal.** The first run had a 2.5 h gap in
the log that made that look obvious, and it was wrong. The second ran under
`nohup` writing to a plain file, finished its downloads in 31 minutes with no
gap, and died identically.

**Cause: `lightkurve/utils.py:558`.** `@suppress_stdout` decorates
`SearchResult.download`/`download_all` and saves/restores the *process-global*
`sys.stdout` around each call. At `_PREFETCH_WORKERS = 4` the save/restore
pairs interleave — thread B saves thread A's devnull as its "original", A
restores the real stdout and its `with open(...)` closes that devnull, then B
restores the closed one. `sys.stdout` stays closed for the rest of the process.
Confirmed by instrumenting the real downloader against MAST and sampling
`sys.stdout` from a watcher thread: **9 transitions, final state closed**.

**The visible crash was the smaller half.** lightkurve logs its quality-mask
line from *inside* `download_all`; once stdout is closed that log call raises
through rich's handler, and `download_one`'s broad `except Exception` records a
**successful** download as `download error`. The first run logged **1,352** of
these starting 17 s in — so its 3,166/4,685 success rate was substantially
spurious, not a property of the data. Nothing was pinned as permanently failed
only because `"I/O operation on closed file"` was already in
`_TRANSIENT_ERROR_MARKERS`.

Fixed by lifting the suppression for the parallel section (`functools.wraps`
exposes the originals on `__wrapped__`) and restoring it after. Re-ran the same
instrumented download: **1 transition, stdout open, zero closed-file errors,
4/4 succeeded.** The regression test drives lightkurve's decorator verbatim
with the interleave forced deterministically. 219 tests green.

**Worth generalising:** a broad `except Exception` around third-party I/O will
happily convert an infrastructure fault into a domain verdict. The download
path reported ~1,200 healthy targets as failures for hours without anything
looking wrong, because "download error" is a plausible thing to see.

State for the next attempt: **3,509 targets cached and ready to score**, 743
cached as permanent "no pipeline data", **433 left to attempt**. Nothing has
been scored yet — `results/candidates_scored.parquet` still does not exist.

## Step 3(b) stage 1 complete — the shortlist, and a bias in it (2026-07-26)

`results/candidates_scored.parquet`: 4,685 TESS PC rows, **3,919 scored ok**,
744 `no_fits` (the genuine "no pipeline data" set), 22 `preprocess_fail`.
37.5 min for the scoring loop. The stdout fix held for the whole run — the
closed-file count stayed at its historical 1,352 from first row to last, and
only 2 download errors were logged, against 1,352 in the two broken runs.

Score distribution over the 3,919: median **0.646**, p90 0.828, p99 0.905,
max **0.9555**. 3,247 ≥ 0.5, 557 ≥ 0.8, **50 ≥ 0.9**, 2 ≥ 0.95. The high median
is expected — every row is already a TFOPWG "PC", so this is a pre-vetted
population, not a blind sample. Note the ceiling: nothing scores above 0.956,
which is Platt saturation on this population, not a model that is ever certain.

**The top of the shortlist is biased toward long-baseline targets, and the
mechanism is not what it looks like.** Candidates with P > 400 d are 1.5% of
the scored set but **6 of the top 20**, with a median of 0.831 and 14.0% at
≥ 0.9 against ≤ 2.4% in every other period band. The obvious reading —
single-transit events with unconstrained catalogue periods — is mostly wrong:
four of those six have 2,200–2,635 d baselines (continuous-viewing-zone
targets) covering 3.6–5.2 transits. Only 2 of the top 20 are effectively
single-transit, and just 66 of 3,919 overall.

Measured across all 3,919, against their actual light curves:

| correlation with prob_mean | Spearman |
|---|---|
| observation baseline (span) | **+0.211** |
| catalogue period | +0.205 |
| **number of transits observed** | **−0.003** |

**The score tracks how long the target was observed, not how many transits
were captured.** n_transits has *zero* relationship with the score. That is
the same effect the injection-recovery control arm found from the other
direction — 26.4% of hosts pass threshold with no injection at all — and it is
now measured on real candidates as well as synthetic ones: the model is
substantially reading host and observation quality rather than transit
repetition. Worth treating as the headline model-behaviour finding to date,
alongside the TESS-vs-Kepler 0.906/0.989 gap.

Practical effect on stage 2: don't filter the shortlist hard, the long-period
entries are mostly well-observed CVZ targets. Do drop the 2 single-transit
ones (TOI 2009.01 / TIC 243187830, 1.72 baselines-per-period; TOI 5725.01 /
TIC 1042432, 1.02) — TRICERATOPS FPP on a period the light curve does not
constrain is minutes per target for little return.

## Stage 0 of the ExoMiner rebuild (2026-07-26)

Commits `edd3715`..`ffe974e`. **Serving still `ca906040` (9-dim) on Fly —
untouched, nothing deployed.** 231 tests green, tree clean.

Direction set by Ollie after reviewing NASA's ExoMiner: rebuild the backend
heavily inspired by ExoMiner++, reorganise the repo in its style, UI redesign
stays the locked final step. Full analysis and staging in
[docs/roadmap.md](docs/roadmap.md); what the model eats today, and what it
does not, in [docs/features.md](docs/features.md).

**The stage-2 shortlist result was not a result.** All 20 targets returned
FPP=0.75 and NFPP=0.0 — identical to twelve decimal places across S/N 3.1–84.5
and 17–1,372 nearby stars. TRICERATOPS' `calc_depths` docstring says ppm but
its arithmetic needs a **fraction** (it computes `tdepth/fluxratio` and zeroes
anything > 1), so ppm zeroed every star, leaving 12 target-side scenarios with
no evidence computed: a uniform 1/12 each, FPP = 1 − 3/12 = 0.75 exactly, NFPP
from a hardcoded branch. Proved on TIC 77175217 — ppm gives 0 of 51 stars
surviving, the fraction gives the target at the right depth. Fixed in
`edd3715`. **`results/candidates_validated.csv` from 2026-07-26 is invalid in
every row** and must be regenerated.

**Ollie supplied a patched TRICERATOPS fork; it is now vendored** at
`pipeline/vendor/triceratops` (MIT, `1.0.20+exohunter.1`). It fixes the NaN the
depth fix exposed next — stock normalisation is `exp(lnZ)/Σexp(lnZ)`, which is
0/0 once the evidences underflow — plus a `log10` background prior added to
natural-log likelihoods (biasing FPP *low*, toward validating planets),
swapped collision masks in the parallel EB paths, and a per-pixel `dblquad`
PSF integral replaced by its exact `ndtr` form. A version pin cannot express
this: stock 1.0.20 satisfies `triceratops>=1.0` and silently returns different
numbers, so `test_vendor_triceratops.py` fails if the env resolves a stock
install. NC-05 is ours — the fork shipped only NumPy shims, so the package
could not import standalone.

**71 GB of staging deleted** (`data/raw/.lightkurve`; 0 of 9,953 manifest paths
pointed into it — pure post-stitch debris) and the cause fixed: staging is now
per-target and removed after its own stitch. It has to be per-target because
`download_many` runs several workers against one cache dir.

**Two landmines deleted.** `preprocess_only.py` wrote 9-dim aux into the same
`views.npz` `build_dataset.py` writes at 13, and was the last script bypassing
`build_labels_from_cfg` — its hand-rolled request had no K2 fields, so running
the *documented recovery script* silently dropped all 530 K2 rows.
`score_target.py` was the last hand-rolled aux implementation, capped at 9
dims. Both fully superseded; single-target scoring is the API or `TargetScorer`.

**The four remaining audit items are closed.** `promotion_gate --models-dir`
resolved the registry's relative path against the caller's cwd; `make
data-push/pull` shelled out to a bare `dvc` (the launchd failure class — and a
`command -v` probe here resolved to a path that does not exist, so every target
that runs project code now goes through `$(PYTHON)`); the flow's `train()` ran
without a data group so MLflow named a full build `cnn-cv-default`; and
`ci.yml`'s promised `catalogue-gate`/`promotion-gate` jobs never existed, which
made refresh_pipeline's "gates are the same code CI runs" false. Both jobs now
run the real scripts against synthetic fixtures the generator validates with
the gates' own schemas; the promotion job asserts both directions.

**Verified end-to-end on a real target.** TIC 451645081 (TOI 783.01, S/N 84.5,
1,372 nearby stars), depth fix + vendored fork, n_draws=200k:

| | before | after |
|---|---|---|
| FPP | 0.75 (constant), then NaN | **0.0258** |
| NFPP | 0.0 (hardcoded branch) | **0.00059** (genuinely summed) |
| best_scenario | *(blank)* | **TP** |
| classification | `likely_fp` | **`likely_planet`** |

The verdict **inverted**. The invalid run was not merely imprecise — it called
a likely planet a likely false positive, and would have done so for anything
the shortlist put in front of it.

### Next: Stage 1 — ExoMiner-grade inputs

Rerun the FPP/NFPP top 20 on the vendored fork first — that closes step 3(b)
with numbers that mean something:

```bash
caffeinate -dis /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/validate_candidates.py \
  --candidates data/labels/candidates.parquet \
  --shortlist results/candidates_scored.parquet \
  --top 20 --out results/candidates_validated.csv --insecure-trilegal \
  2>&1 | tee outputs/validate-candidates.log
```

Skip TOI 2009.01 (TIC 243187830) and TOI 5725.01 (TIC 1042432) — effectively
single-transit, so FPP on an unconstrained period buys little. Read NFPP as the
headline and FPP as directional.

Then Stage 1 proper: the new view set (301/31 with variance channels, odd/even,
secondary, centroid, trend, unfolded per-transit, momentum dump, periodogram
pair), TESS DV XML ingest for difference images and DV scalars, Gaia RUWE, and
the FFI fallback for part of the 744 `no_fits`.

## Step 3(b) closed — the validated shortlist (2026-07-31)

Commits `edd3715`..`1e45493`. **Serving still `ca906040` (9-dim) on Fly —
untouched throughout.** 240 tests green.

`results/candidates_validated_final.csv` is the artefact: TRICERATOPS FPP/NFPP
for the top 20 candidates by model score, at `n_draws=1e6` (recorded per row).

| verdict | n |
|---|---:|
| validated_planet | 1 |
| likely_planet | 6 |
| inconclusive | 3 |
| likely_nearby_fp | 3 |
| likely_fp | 1 |
| degenerate (no result) | 2 |
| timeout / error | 4 |

**TOI 5112.01 (TIC 337385330) reached FPP 0.0136** against the 0.015 threshold,
NFPP **2.1e-63** against 1e-3, at S/N 31.3 (above the S/N 15 floor where FPP
becomes unreliable) — on a single TRILEGAL draw. **The re-run on 2026-08-01
withdrew the validation; see below.**

**Four `NEBx2P`/`BEBx2P` verdicts are the validation layer earning its keep.**
A neighbouring or background eclipsing binary at twice the period is invisible
to a light-curve-only CNN; all four scored above 0.918 in the model.

### The four bugs that had to be fixed to get here

Every one of these produced *confident-looking wrong numbers*, not crashes.

1. **`calc_depths` takes a fraction, not ppm** (`edd3715`). Its docstring says
   ppm; its arithmetic computes `tdepth/fluxratio` and zeroes anything > 1.
   Fed ppm, every star zeroed, leaving 12 scenarios with no evidence computed —
   uniform 1/12, so FPP was **exactly 1 − 3/12 = 0.75 for all 20 targets**.
   The whole first shortlist was that constant.
2. **Stock TRICERATOPS is numerically unsound here** — vendored fork at
   `pipeline/vendor/triceratops` (`b5f71c6`). `exp(lnZ)/Σexp(lnZ)` is 0/0 once
   the evidences underflow, which is the NaN FPP the depth fix exposed next.
   Also fixes a `log10` background prior added to natural-log likelihoods
   (biasing FPP **low**, toward validating planets) and swapped collision masks
   in the parallel EB paths. A version pin cannot express this: stock 1.0.20
   satisfies `triceratops>=1.0` and returns different numbers, so
   `test_vendor_triceratops.py` fails if the env resolves a stock install.
3. **A uniform posterior is not a verdict** (`e7ed3a8`). TIC 441804533 returned
   FPP 6/7, NFPP 2/7 — 21 scenarios at exactly 1/21 — and was reported
   `likely_nearby_fp`. The fork's own `FPP_degenerate` does **not** catch this:
   it flags -inf/NaN evidences, but a uniform *finite* posterior normalises
   cleanly and reports "ok". Both checks now run, plus a non-finite guard.
   Two of the 20 are `degenerate`, both because the TIC lacks stellar mass,
   radius, Teff and parallax — a catalogue gap, not a numerical one.
4. **TRILEGAL is stochastic and we re-queried it every run** (`13b0183`). It is
   a Monte Carlo galaxy simulation; its star count feeds `lnprior_background`
   directly. Two runs of TIC 468983280 disagreed — `BEBx2P` with NFPP 2.4e-10
   versus `NEBx2P` with NFPP 0.9995, flipping the verdict — on identical code.
   `--trilegal-cache` (default `data/trilegal/`) pins one population per target.
   It also cut per-target runtime from 15–22 min to 1–6 min: that query was
   most of the wall time.

### Operational lessons

- **Three targets are intractable at `n_draws=1e6`** — TIC 451645081, 234345288
  and 300710077 each burned a 2 h cap, and 451645081 once ran 10+ h. Raising
  the cap does not help. They need `--n-draws 200000`; `n_draws` is recorded
  per row so reduced precision is visible rather than silently pooled.
- **`--skip-completed` exists now** because the docstring claimed
  "resumable-by-rerun" while nothing skipped completed work; one interruption
  cost ~3 h redoing ten finished targets. `error`/`timeout` rows are retried,
  `degenerate` counts as done.
- **Use absolute paths.** Run from the wrong directory and both Python and
  `tee` fail to stderr, so the run silently does not happen and leaves no trace.
- The rich log handler eats `[validate]` as markup — grep `TIC [0-9]+`, and note
  failures read `TIC 123 failed:` with no colon after the number.

**Outstanding:** the four unfinished targets are being retried at 200k draws
into the same file. TIC 1042432 will probably fail again — its TRILEGAL query
returns something unparseable for those coordinates, which fewer draws cannot fix.

## TOI 5112.01 is not validated — the FPP straddles the threshold (2026-08-01)

The 200k-draw retry of the four unfinished targets **resolved none of them**:
451645081, 234345288 and 300710077 still time out at the 3600 s cap with no
`n_draws` recorded, and 1042432 still fails on its TRILEGAL query. The file
stands at 20 rows, **14 usable**. Fewer draws was the wrong lever.

Three fresh runs at `n_draws=1e6` with `--trilegal-cache ""`, so TRILEGAL was
the only variable (`best_scenario`, `n_nearby_stars=156` and `snr=31.29` are
identical across all four draws, which is what makes this a clean experiment):

| draw | TRILEGAL stars | FPP | NFPP | verdict |
|---|---:|---:|---:|---|
| original | *(cached)* | 0.0136 | 2.1e-63 | validated_planet |
| rerun 1 | 384 | **0.0125** | 8.0e-64 | validated_planet |
| rerun 2 | 394 | **0.0175** | 1.0e-63 | likely_planet |
| rerun 3 | 370 | **0.0170** | 1.5e-63 | likely_planet |

**Two of three fresh draws land above the 0.015 threshold**, and the mean
(0.0157) is above it too. Spread is 0.0125-0.0175, std 0.0028 — **18% of the
mean**, against the 10% the earlier note guessed would be enough to cross. The
threshold sits in the middle of the sampling distribution, so which side a
single run reports is close to a coin flip. Artefact:
`results/toi5112_stability.csv`.

**A 6% swing in TRILEGAL star count (370-394) moved FPP by 18%**, so the
background prior is the dominant term here, not the light curve.

**NFPP is rock solid** — 8e-64 to 2e-63 across every draw, ~60 orders of
magnitude below its 1e-3 threshold. That is the expected split: TRILEGAL
simulates the *background* population and feeds `lnprior_background`, so it
moves BEB scenarios and leaves NEB alone. The nearby-false-positive case
against TOI 5112.01 is genuinely closed; the total FPP is not.

**Read it as a strong candidate, not a validated planet.** Anything public
needs either many draws averaged into a distribution rather than a point
estimate, or an independent line of evidence. One TRILEGAL draw is not a
verdict — the same lesson as `13b0183`, one threshold further in.

## Stage 1 — DV ingest landed, view set foundations built (2026-08-01)

**Serving still `ca906040` (9-dim) on Fly — untouched, nothing deployed.** 286
tests green (was 240), ruff and mypy clean on the new modules. Not yet
committed: one push per stage.

### The DV archive

`data/raw_dv/` — **5,766 targets fetched, 1,438 with no DV products, 0
outstanding failures, 3.6 GB** over **7,204** TESS TICs. Coverage **80.0%**,
within a point of the 81.5% measured on a 200-target sample beforehand. 5.3 h
wall clock. Zero filename-parse warnings across the whole corpus.

The target set is 7,204 rather than the 7,199 sized on 2026-07-31 because **the
scheduled refresh ran at 09:00 on 2026-08-01, during the DV pull**, growing the
label catalogue 5,686 -> 5,703 rows (+17 TESS). The sweep re-run afterwards
picked the new targets up, so the archive is complete against the current
catalogue, and all six gates — including `label-shrink` and `refresh-leakage`
against `labels.previous.parquet` — pass.

Audit of 341 reports from 300 random targets: **0 parse failures**, every XML's
`ticId` matches its directory, 100% carry at least one difference image (1,613
in the sample, median 2 per report, max 43). MES, bootstrap significance, ghost
core statistic and difference-image quality are present in 100%; stellar mass
is derivable from density and radius in 100%.

### Three things that change Stage 2

**Difference-image stamps are 11, 13, 15 or 17 px — not a fixed 33x33.** That
is Kepler's size. The extent is the target's aperture, and the pixel list is
sparse with absolute `ccdRow`/`ccdColumn`. Stage 2 must re-grid to a fixed
stamp; the parser deliberately returns the sparse arrays and a bounding box
rather than inventing a shape.

**~9% of targets' DV diagnostics belong to a different signal.** Over 240
labelled targets with a TCE: 90.4% match the catalogue period to within 1%,
0.4% within 1-10%, and **9.2% are off by 10% or more**. About 5% are clean
harmonics (2.9% double, 1.7% half, 0.4% triple); the rest are unrelated
long-period TCEs SPOC found instead — ratios of 18x, 24x, 46x, 94x. A report
holds one `planetResults` per TCE, so taking the first would attach the wrong
transit's diagnostics to our row and nothing downstream would notice.
`parse_dv_xml` matches on the catalogue period and returns
`period_mismatch_frac`; **that field must gate the DV branch presence mask, not
just emit a log line.**

**DV's own transit counts confirm the completeness thesis.** Median
observed/expected ratio is **0.29**, and only **12%** of reports caught every
transit their ephemeris predicts over the baseline. Two targets with identical
folded views can differ by a factor of three in how much real transit evidence
they contain — invisible in a fold, which is the point of the unfolded branch.

### Sizing corrections to the roadmap

- **Sector scope was already on disk.** Not the download manifest — that stores
  `n_sectors` as a count only, and the stitched FITS keeps just the last
  sector's header. It is the `sectors` column of
  `data/catalogue/candidates.parquet`: 7,195 of 7,199 covered, zero queries.
- **14-56 GB was ~5x too high.** DV XML is ~0.34 MB, not 2-8 MB — that figure
  was the DVR *PDF* (18-21 MB) and DVT *FITS* (11-22 MB). Actual: 3.5 GB.
- **Batching matters more than sector scoping.** `query_criteria` accepts a
  list of `target_name`s; 40 per round trip is 0.29 s/target against 1.8 s.

25,429 products were skipped by the selection policy (widest multi-sector run
per target) and are recorded in the manifest with size and URI, so widening to
every run is a download-only pass with no second availability sweep.

### Code

- `data/dv.py` — batched availability query, resumable fetch, manifest mirroring
  `LightCurveDownloader` (atomic writes, permanent failures cached, transient
  ones never pinned). 82 transient failures on the first pass were correctly
  left uncached and swept by a re-run.
- `data/dv_xml.py` — the parser. Three bugs found by checking a real file
  rather than trusting plausible output: `sectorsObserved` is indexed *directly*
  by sector (position 0 unused; an off-by-one mislabelled every difference image
  by a sector, caught by cross-checking against `differenceImageResults/@sector`);
  `weakSecondary` hangs off `planetCandidate`, so looking under `planetResults`
  returned None for every target; and `-1.0` sentinels now become None instead
  of entering aggregates as measurements.
- `preprocess/viewset.py` — 301/31 views as `[flux, scatter, present]`, plus
  odd/even, secondary, trend and the unfolded `[20,31]` branch with
  observed/expected counts. **The comparison views share the primary's depth
  scale.** Normalising each by its own depth sent odd, even, secondary and every
  unfolded transit to exactly -1.000 — which looks entirely reasonable and
  deletes the three diagnostics they exist to provide. On TIC 337385330 the fix
  gives odd -1.135 vs even -0.714, secondary -0.257, per-transit depths from
  -0.58 to -1.43.
- `preprocess/fold.py` — `bin_profile` returns median, MAD scatter and count in
  one sort-and-split pass, replacing an O(bins x points) per-bin mask. The
  median path is pinned against the original implementation as an oracle,
  because `fold_and_bin` feeds live serving.
- `validation/schemas.py` — `check_dv_archive` as a sixth gate. Its headline
  check is presence-mask integrity: a target never queried is indistinguishable
  from one queried with no DV products, so an interrupted fetch would silently
  mask out real data for everything after the interruption.

### Still open in Stage 1

TESS-SPOC FFI fallback for the 744 `no_fits` candidates; wiring the new view set
into `build_dataset.py` and a shard schema (measure shard size after a few
hundred targets, and revisit `.cache()`); momentum-dump, periodogram-pair and
centroid branches; extending the views gate to the new artefact.

### DV scalars table and Gaia RUWE (2026-08-01, later)

**`data/processed/dv_scalars.parquet`** — 3.6 GB of XML reduced to **1.9 MB**,
6,484 rows over 5,766 targets, **0 parse failures**. Built by
`scripts/build_dv_table.py`, which needs no network. Difference-image *pixels*
stay in the XML until stage 2 fixes a stamp size; this is the scalar half.

The column that matters is **`dv_usable`**: 92.9% true, **452 rows (7.0%) false
because the best-matching TCE is a different signal** than our catalogue row,
and 8 unverifiable (no catalogue period). Mask on it. Training the DV branch on
those 452 would attach another transit's bootstrap FAP, ghost statistic and
transit counts to our candidate, and nothing downstream would flag it.

**`data/gaia/ruwe.parquet`** — 7,177 of 7,204 targets (99.6% have a Gaia
counterpart), **7,071 with RUWE (98.5%)**. Median **1.028**; **16.5% (1,166
targets) above the 1.4** unresolved-binary cut, which is a large enough slice to
matter as a feature rather than a footnote.

Two hops, because the catalogues do not line up: TIC v8 carries a Gaia **DR2**
source id, and `ruwe` is a **DR3** column, so it routes through
`gaiadr3.dr2_neighbourhood`. That table is many-to-many — **7.0% of our targets
(499) have more than one DR3 candidate**, i.e. DR3 resolved a blend DR2 saw as
one source. `_best_match` keeps the nearest and records `n_dr3_candidates`;
taking an arbitrary row would have attached a *neighbour's* RUWE to the target,
and a neighbour's RUWE is a perfectly plausible number.

Both artefacts are DVC-tracked (pointers in git, bytes need `dvc push`).

## Stage 1 complete — the view set is built and gated (2026-08-05)

**Serving still `ca906040` (9-dim) on Fly — untouched throughout.** 300 tests
green (was 240), ruff and mypy clean, all **seven** gates pass against the real
artefacts.

### The artefacts

| | |
|---|---|
| `data/raw_dv/` | 5,766 targets, 3.6 GB, 80.0% coverage |
| `data/processed/dv_scalars.parquet` | 6,484 rows, 1.9 MB, 0 parse failures |
| `data/gaia/ruwe.parquet` | 7,071 with RUWE, median 1.028, 16.5% above 1.4 |
| `data/raw_ffi/` | FFI light curves for the `no_fits` candidates |
| `data/processed/viewset.npz` | **5,423 examples, 65 MB** |
| `data/processed/viewset_tfrecords/` | **11 shards, 15 scalars, 2 masks, 122 MB** |

5,423 of 5,703 labelled targets built (95.1%): 273 have no cached FITS, 7 no
ephemeris, and **zero preprocess errors**. Sources: Kepler 2,500, SPOC-2min
2,392, K2 527, FFI 4.

**Shard size is settled: 122 MB against the legacy 47 MB — 2.6x, not the 20-50x
the roadmap allowed for.** `tf.data.cache()` needs no revisiting.

**The presence masks are doing real work.** `dv_usable` is 87.4% on TESS and
**0% on Kepler and K2**; RUWE the same shape. That is the "a missing branch
poisons every row of its mission" case being stated explicitly rather than
imputed as a zero.

### The bug that passed every gate

The DV scalars table publishes its **own** observed/expected transit counts.
The side-table merge left both unrenamed, so pandas suffixed ours *and* theirs
to `_x`/`_y`, and the shard writer's `if c in scalars.columns` filter then wrote
**13 scalars instead of 15 without a word**.

The two columns lost were `observed_transit_count` and `expected_transit_count`
— the exact pair the unfolded branch exists to provide, and the pair stage
2(b)'s success criterion is measured on. The build succeeded, all seven gates
passed, and the shards were wrong. Caught only by counting the scalars in
`metadata.json` against `FEATURE_COLUMNS`.

Fixed twice over: DV's counts are prefixed `dv_*` and a guard raises on any
side-table collision, and the shard writer now logs every declared column it
cannot find. Silently writing a shorter vector is what let this reach disk.

### FFI recovery works

The 744 `no_fits` candidates are targets SPOC never produced a 2-minute light
curve for — not badly scored, **absent**. A 40-target probe found **100%**
recoverable from an FFI author (QLP 100%, TGLC 75%, TESS-SPOC 42.5%). The view
builder was then tested on 12 real FFI curves at 200 s and 600 s cadence:
12/12 build finite views with sensible transit counts.

They are cached separately (`data/raw_ffi/`) and never mixed into `data/raw/`,
because FFI cadence is 200 s to 30 min against SPOC's 120 s.

### The momentum-dump branch could not be built as specified

TESS flags reaction-wheel desaturations in `QUALITY` bit 5, but lightkurve's
default `quality_bitmask` drops those cadences at download — **bit 32 is set on
zero cadences across the entire cache**, because the flagged rows are not in the
files. Reading it would have shipped an all-zero branch for the whole corpus
that looked like a working feature.

It measures the *hole* a dump leaves instead. That needed segmenting by
observation gap: measuring against the full baseline counted the multi-year
holes between sectors and pinned every bin at ~87% with no discriminating power.
Segmented, it reads 0-2% typical with real peaks to 17%.

### Also corrected

`sectorsObserved` is indexed **directly by sector** (position 0 unused), not
offset by one — cross-checked against `differenceImageResults/@sector`, since an
off-by-one would mislabel every difference image by a sector.
`weakSecondary` hangs off `planetCandidate`, not `planetResults`; looking in the
wrong place returned None for every target. `-1.0` sentinels are now None.

**Difference-image stamps are 11/13/15/17 px, not a fixed 33x33** — that is
Kepler's size. Stage 2(d) must re-grid.

**~9% of targets' DV diagnostics describe a different signal** (90.4% match the
catalogue period within 1%; ~5% are clean 2x/0.5x harmonics, the rest unrelated
long-period TCEs). `dv_usable` masks them.

## Stage 2(a) run 1 — rejected, and the premise needs re-examining (2026-08-05)

**Serving still `ca906040` — untouched. Nothing promoted; the registry is
unchanged.** Artefacts in `models/cv/branches-20260805/`.

### The gate rejects it

| | stage 2(a) branches | incumbent `ca906040` |
|---|---:|---:|
| ROC-AUC | **0.9325 ± 0.0059** | **0.9581 ± 0.0057** |
| PR-AUC | 0.9308 | 0.9599 |
| Brier | 0.1003 | 0.0791 |
| ECE | 0.0301 | 0.0276 |

`promotion_gate` → REJECT. Not a like-for-like comparison: the incumbent is
Optuna-tuned with augmentation on 2001/201 views; the branch model is a first
pass at 193k params, 2 conv blocks, no augmentation, no tuning. How much of the
2.6-point gap is architecture and how much is tuning is not yet known.

### The +0.211 baseline correlation is label structure, not a model pathology

This is the finding that matters, and it challenges the roadmap's premise.

Measured on the labelled CV set (5,423 rows), with **baseline in days** — the
covariate the original figure used:

| | baseline (d) | period | n_transits |
|---|---:|---:|---:|
| incumbent `ca906040` | +0.238 | +0.287 | −0.087 |
| stage 2(a) branches | +0.239 | +0.349 | −0.116 |
| **ground-truth label** | **+0.278** | **+0.265** | **−0.073** |

**The labels themselves correlate +0.278 with observation baseline** — and on
TESS alone, **+0.387**. Both models sit *below* that. Neither is over-reading
baseline; both slightly under-read a relationship that is really in the data.

It is not a period artefact. Inside TESS period bands the correlation holds at
+0.408 (<3 d), +0.384 (3-10 d), +0.270 (10-30 d), +0.226 (>30 d). And the
medians are stark: **TESS confirmed planets have a median baseline of 1,495 d
against 430 d for false positives**, a 3.5x difference.

The mechanism is almost certainly **confirmation bias in the catalogue**: a
target observed across many sectors accumulates follow-up and gets promoted to
confirmed, while a briefly-observed one stays a candidate or is retired as a
false positive. The model learns "long baseline -> likely confirmed" because in
the training labels that is true.

### Two measurement errors in the original framing, both mine to flag

**Population.** The +0.211 / −0.003 figures were measured on **3,919 scored
candidates**, not the labelled set. Candidates have no labels, so the
label-structure comparison above cannot be made there. The two populations are
not interchangeable and the roadmap does not distinguish them.

**Covariate.** `observation baseline` in the original is a span in **days**. An
earlier version of `eval/observation_bias.py` proxied it with
`expected_transit_count`, which is baseline / period — a different quantity that
partly cancels the effect being measured. Fixed; the table above uses days.

### What this means for stage 2(b)

Its success criterion is *"corr(prob, n_transits) must leave zero **and** the
26.4% control-arm host-pass rate must fall"*. On the labelled set, driving the
baseline correlation toward zero moves the model **away** from the label
structure (+0.278), which should cost AUC without making it more truthful.

If the effect is a selection artefact in the labels, no architecture change
fixes it — that is **stage 3** (labels and negatives), not stage 2. The
criterion needs re-specifying before stage 2(b) is worth running, and that is a
call for Ollie.

**Not yet tested:** the candidate population, where the original finding lives.
That needs the branch model scored over the 4,685 candidates, which needs
candidate views built — the same `build_viewset.py` run against
`candidates.parquet` rather than `labels.parquet`.

## The candidate-population recomputation — criterion retired (2026-08-05, later)

**Serving still `ca906040` — untouched. Nothing promoted.** New artefacts:
`results/candidate_observation_bias.json`, `pipeline/scripts/candidate_bias.py`.

### The number, and Ollie's rule applied

Incumbent scores over the candidate population, baseline as a span in **days**:

| | |
|---|---:|
| candidate baseline correlation | **+0.208** (n=3,908) |
| the same, controlling for period | +0.187 |
| label correlation, all-mission | +0.278 |
| **label correlation, TESS** | **+0.387** |

Every scored candidate is TESS — `candidates_scored.parquet` is TESS-only, and
the 698 Kepler candidates in the view set were never scored by the incumbent —
so **+0.387 is the comparison that matters**, not the all-mission +0.278.
Comparing a TESS-only measurement against an all-mission reference would have
been the third population mismatch in this sequence.

+0.208 against +0.387 is *at or below* the label structure, by a wide margin.
The pre-committed rule's second branch fires: **the "baseline to zero" criterion
is retired**, 2(b) runs on AUC and calibration, and observation selection moves
to stage 3, written up there as a real problem architecture cannot fix.

### Which figure carried the error — it was not the one we expected

The instruction was to identify the original figures rather than assume they
shared the module's covariate bug. Both covariates were computed on the same
rows:

```
baseline_days            +0.2075   <- the roadmap's "+0.211"
expected_transit_count   -0.0025   <- the roadmap's "-0.003"
observed_transit_count   -0.0476
```

**The +0.211 was correct all along.** It reproduces to within 0.003 with
baseline in days, so the suspicion recorded above — that it might carry the same
proxy error — is disproven.

**The −0.003 is the one that was wrong.** It was measured against
`expected_transit_count`, the transits the ephemeris *predicts*. The roadmap
describes it as "how many transits were caught", and against transits actually
captured the figure is **−0.048**. The qualitative claim survives (the score
still barely tracks captured transits); the number in the roadmap was measuring
a different quantity.

`expected_transit_count` is baseline / period. The original analysis used it as
a *transit count*; `eval/observation_bias.py` later used it as a *baseline
proxy*. Two contradictory roles for one column, erring in opposite directions,
which is why neither looked wrong on its own.

### The covariate fix had never reached the code

`observation_bias.py` still defaulted `baseline_column="expected_transit_count"`
with a docstring endorsing it, and `train_branches.py` called it with no
override. **The committed `models/cv/branches-20260805/observation_bias.json`
therefore holds the uncorrected numbers** (`baseline_sensitivity: -0.068`); the
+0.239 in the section above was computed ad hoc in a session and never written
down as code. The next CV run would have produced another confident wrong number.

`baseline_days` is now a named, derived, tested covariate:
`(expected_transit_count - 1) * period`, which is the epoch span, quantised to
whole periods and floored at zero when only one transit is predicted.
`test_baseline_defaults_to_days_not_the_transit_count` pins the default so it
cannot revert silently.

### Two checks that were worth running

**The reconstruction is sound.** `rho(baseline_days, n_sectors_observed)` =
**+0.642** over 3,642 targets, against a DV-sourced sector count that shares no
arithmetic with our ephemeris. Not higher because the two measure different
things: span counts the multi-year gaps between sectors, sector count does not.

**It is not a period artefact.** `rho(baseline_days, period)` is only +0.142, and
the partial correlation controlling for period is **+0.187** against a raw
+0.208. It also survives inside every period band (+0.146 <3 d, +0.169 3–10 d,
+0.254 10–30 d, +0.062 >30 d, the last on n=160). This matches the labelled-set
behaviour.

Also worth recording: `rho(prob, n_sectors_observed)` = +0.222, partialling to
+0.184 — statistically indistinguishable from the baseline effect. Span and data
volume cannot be separated on this population, so "more time to accumulate
follow-up" and "more data to detect with" remain confounded. A stage 3
intervention should not assume it is only the former.

### Stage 2(b)'s criterion, split rather than deleted

Its two halves were never the same kind of measurement, and only one was
label-confounded:

- **The 26.4% control-arm host-pass rate is now the criterion.** It is measured
  on real hosts with *no injection*, so no label structure enters it. It stands
  unchanged and must fall.
- **The baseline correlation is a reported diagnostic, not a gate.**
- **The transit-count correlation is reported, not gated** — its zero point is
  −0.048, and the labels sit at −0.073, so there is no defensible target.

## Audit regression sweep + Stage 2(a) close-out (2026-08-06)

**Serving still `ca906040` (9-dim, 2001/201) on Fly — untouched. Nothing
promoted; the registry is unchanged.** 401 tests green (was 304), ruff and mypy
clean, seven data gates pass.

### The sweep — every finding from both prior audits

Audit records re-checked: **2026-07-13** (nine targets + two follow-ups) and
**2026-07-25** (`1431660`..`c41fa62`, security + V1-remnants). Each fix was
looked for *in the module it belongs to*, not at a call site.

| finding | still closed? | test pinning it |
|---|---|---|
| Kepler subsample churn — positional sampling | yes, `catalog._stable_sample` (md5 of `seed:tic_id`) | `test_selection_stable_under_realistic_pool_growth` |
| Machine-specific paths | yes, `run-api.sh` discovers the env, `$EXO_PYTHON` overrides | — (shell) |
| Docs drift | yes | — |
| `/score` latency (unbounded BLS) | yes, ephemeris user > catalogue > BLS, grid capped | `test_search.py` |
| Serving calibration mismatch — scored weights ≠ shipped file | yes in `train.py:396` ("score what ships"); **REGRESSED in `train_branches.py`** — see below | now `test_every_fold_leaves_a_servable_artefact` |
| Credentials in the build context | yes, `.dockerignore:28` excludes `.dvc/config.local` | — |
| `inf` / unbounded ephemeris + `tic_id` | yes, bounds reject non-finite, `t0` at 1e8 keeps the three malformed CTOIs scoring | `test_score_rejects_non_finite_ephemeris`, `test_score_accepts_worst_real_catalogue_ephemeris` |
| Response-cache race + 128 cap | yes, `_cache_lock` / `_CACHE_MAX` | `test_score_cache_is_bounded_and_thread_safe` |
| 404 leaked server paths | yes, `_redact_paths` at the boundary | `test_score_404_keeps_path_free_detail` |
| `render_vetting` read ExoFOP's raw header | yes, reads `disposition` | — |
| Test-hygiene: shared response cache across tests | yes, autouse fixture clears it | — |
| `preprocess_only.py` / `score_target.py` could write 9-dim/no-K2 data | yes — both deleted in stage 0 | `test_imports.py` |
| Promotion gate `--models-dir` half-honoured | yes, resolves against `models_dir.parent` | `test_incumbent_summary_resolves_registry_path_from_any_cwd` |
| `dvc` as a bare command | yes, `Makefile` uses `$(PYTHON) -m dvc` | — |
| CI missing the gate jobs its comment promised | yes, `catalogue-gate` + `promotion-gate` jobs exist | — |
| Observation-baseline covariate (the precedent) | yes, `BASELINE_DAYS` is the default and is derived | `test_baseline_defaults_to_days_not_the_transit_count` |

**One regression, and it is the same class as the precedent.** The 2026-07-14
fix "reload the checkpoint before scoring, because in-memory weights after
`fit()` are not the file that ships" lives in `train.py` and **was never carried
into `train_branches.py`**. Worse than a drift: that trainer wrote **no
checkpoint at all**. Stage 2(a) run 1 scored weights that existed only in
memory, and `models/cv/branches-20260805/` has no `fold_*` directories — the
run cannot be rescored, recalibrated, promoted or served. Fixed in the module:
per-fold `ModelCheckpoint` + reload before scoring, plus a calibrator bundle
carrying the Platt fit and the fold's scalar constants.

Fixing it immediately exposed a second defect: **the branch model could not be
deserialised at all.** Gating used a `Lambda` over a Python lambda, which Keras
refuses to load without `safe_mode=False`. Replaced with registered
`PresenceFlag` / `PickColumns` layers, so a promoted checkpoint loads without
waiving a safety check.

Still open by decision, verified unchanged and *not* regressions: the
9-dim/13-dim script crashes, `force_download` on the HTTP surface, the
`_score_lock` timeout, `/candidates.csv` row cap, and `/healthz` trusting
`registry.json` alone.

Housekeeping applied this session: `chmod 600 .dvc/config.local` (was 0644),
`DVC_NO_ANALYTICS=1` in `docker/api.Dockerfile`, and the "fetch all branches
first" warning added to the `dvc gc` recipe in `docs/OPERATING.md`. Not checked
because it is not visible from here: the Cloudflare R2 Public Development URL.

### Security pass since `c41fa62` — 39 commits, clean

Zero blobs over 1 MB entered git; every new data artefact is a `.dvc` pointer.
No secret-shaped string in any added line. Every new file lands in
`pipeline/`, `api/`, `docs/`, `.github/` or as a DVC pointer. The only new
network calls are `astroquery` to MAST and Gaia inside the fetch modules, and
no source file carries an absolute path.

### K2 — 9.7% of training, benchmarked for the first time

The incumbent's 4,818 out-of-fold rows contain **zero K2**, so every comparison
against it inner-joined all 527 rows away in silence:

```
in_shared    False   True    All
K2             527      0    527
Kepler         262   2238   2500
TESS            29   2367   2396
```

The legacy shards do carry all 527 at 2001/201, so it was scorable —
`pipeline/scripts/benchmark_incumbent_k2.py`,
`results/incumbent_k2_benchmark.json`:

| K2, n=527 (315 pos) | incumbent | branches |
|---|---:|---:|
| ROC-AUC | **0.9348** | 0.9189 |
| PR-AUC | 0.9470 | 0.9176 |
| Brier | 0.1538 | **0.0957** |
| ECE | 0.1989 | **0.0500** |
| recall @1% FPR | **0.190** | 0.089 |

**Not like-for-like, and the asymmetry favours the branch model.** No K2 row
was in any of the incumbent's training folds, so its numbers are zero-shot
cross-mission transfer scored by all five checkpoints; the branch model's are
ordinary out-of-fold with K2 in four folds of five. The incumbent still wins
ranking by 0.0159. Its Brier and ECE are much worse and should not be read as a
branch-model win: the Platt scalers were fitted on Kepler+TESS validation rows
and K2's base rate is 0.598.

Two rebuild details that would otherwise have produced a confident wrong number.
**The 9-dim and 13-dim aux layouts disagree at index 7** — catalogue SNR versus
`pink_snr` — so slicing 13 → 9 would feed the model a different feature in a
lane it learned as something else; the vector is rebuilt instead. And
**catalogue SNR is absent on all 527 K2 rows**, so that lane imputes to the
training median, the same path a non-TOI takes at serve time.

`eval/comparison.py` now computes per-mission coverage first and names any
mission an inner join drops entirely (`MissionCoverage.dropped`,
`compare_prediction_sets(strict=True)`). It reproduces every recorded stage
2(a) number from the artefacts on disk: +0.0222 all-mission on 4,605 rows,
Kepler +0.0348, TESS +0.0021, weights 48.6 / 51.4 / 0.

### Recall @1% FPR is now a gate criterion

AUC scores ranking at every threshold; a shortlist lives at one. On TESS the
two models are 0.002 apart on AUC and **0.069 apart on recall @1% FPR**
(0.307 incumbent, 0.238 branches). `evaluate_promotion` now rejects on a
shortlist-recall drop beyond 0.02, and gates on the **TESS** slice with Kepler
and K2 as alarmed diagnostics and the all-mission aggregate reported but never
gating. Summaries without a `per_mission` block — every run before this one,
including the live incumbent — fall back to pooled means, as the ECE guard does.

### Augmentation for the view set

Run 1 trained without augmentation against an incumbent that had it. `augment_views`
takes two `(bins, 1)` tensors and cannot be reused, so `datasets/viewset_augment.py`
reimplements its semantics per view — same ops, same magnitudes, same order.

Two things the view set changes. **The presence channel is never augmented**:
`_gated` reads `reduce_max(present) > 0`, so noise on an absent branch's zeros
flips it to "present" and disables the gating for the 3,027 of 5,423 rows where
`dv_usable` is 0%. Noise is also multiplied by presence, so an unmeasured bin
keeps the zero that says so. **Not every axis is phase**: the periodogram views
are period-indexed and take no phase shift, and the unfolded stack shifts within
each transit rather than reordering transits. `VIEW_KINDS` declares which is
which and raises on a view that has no entry.

### The resolution fix — running, pre-registered

`GLOBAL_BINS`/`LOCAL_BINS` restored to **2001/201**. Measured before launch:
**233,617 parameters** (from 192,817), **~669 MB of shards** (126.4 KB/example,
×5.5 on 122 MB), **~5.4 GB peak RSS** — of which ~4.7 GB is fixed TF/Metal cost,
measured by running the same fold at 301/31 on the full shard set (4,861 MB).

Note the confound rather than bank it: 233,617 lands *above* the incumbent's
227,641 and above the 226,711 of the cancelled capacity run, so a Kepler gain
cannot be attributed to resolution alone.

Two smaller fixes the change forced, both of the "declared twice" family that
has bitten this project repeatedly. `VIEW_SHAPES` now **derives** from the
builder's constants instead of restating them, so resolution is one edit. And
the per-target interim cache is **keyed by resolution** (`g2001l201/`), so a
rebuild cannot read back arrays at the old bin count — the old 301/31 cache is
still on disk and still valid.

The pre-registered reading of the result is in `docs/roadmap.md`, written
before the run finishes.

### Where run 2 stopped, and how to resume it

Stopped deliberately at **889 / 5,703** targets built (all 889 verified
readable). The build is idempotent and resumable — `_cache_path` skips anything
already cached, so this picks up where it left off:

```bash
cd /Users/ollie/Project/v2
nohup /opt/anaconda3/envs/exoplanet-hunter-v2/bin/python \
  pipeline/scripts/build_viewset.py > outputs/build-viewset-2001.log 2>&1 &
```

Measured rate **~49 targets/min**, so the remaining ~4,800 take **~100 min**.
Then, in order:

```bash
/opt/anaconda3/envs/exoplanet-hunter-v2/bin/python pipeline/scripts/shard_viewset.py
/opt/anaconda3/envs/exoplanet-hunter-v2/bin/python pipeline/scripts/train_branches.py
```

Expect ~669 MB of shards and ~5.4 GB peak RSS during CV. The old 301/31 interim
cache (5,423 files, `data/interim/viewset/*.npz`) is untouched and still valid,
so reverting the two constants restores run 1's inputs without a rebuild.

**If the resumed build fails loading a `.npz`**, one file was truncated by the
stop: delete that file and re-run. A sweep of all 889 found none.

**Read the result against the pre-registered rule in `docs/roadmap.md`, not
against hope.** Kepler gap under ~0.012 with TESS level means resolution was the
cause; above ~0.020 falsifies it and makes the capacity run mandatory. Nothing
is promoted without the gate, and the gate now decides on TESS.

Not yet done, and Ollie's: `git push`, `dvc add` + `dvc push` for the rebuilt
view set and shards, `fly deploy` (nothing needs deploying — serving is
unchanged).

### A NaN metric promoted — found by pre-flighting the CV path (2026-08-06)

Running the new CV entry point end to end on a 296-target probe (deliberately
single-class, so its AUC is undefined) returned:

```
PROMOTE: ... ROC-AUC nan vs incumbent 0.9581; Brier 0.0000 vs incumbent 0.0791
```

**Every guard in `evaluate_promotion` is an inequality, and NaN loses all of
them.** `nan <= 0.9581` is False, so the AUC check passes; `nan > inc + tol` is
False, so the Brier and ECE checks pass. Any degenerate run — a single-class
fold, an empty mission slice, a blown-up loss — would have promoted itself with
a summary that says `nan` in plain text.

`evaluate_promotion` now checks every gating metric is finite *before* any
comparison and rejects with "not measurable on this run: ...". Pinned by
`test_a_nan_metric_rejects_instead_of_sailing_through_every_guard` and a
parametrised case over all four gating metrics.

This is the third instance of the same failure mode in this project: a check
that returns a plausible answer instead of failing. It was only visible because
the run was executed rather than reasoned about.

### Pre-flight of run 2's CV path — clean

`pipeline/scripts/train_branches.py` was exercised end to end at 2001/201 before
committing to the long run. It writes per-fold `cnn_branches.keras` +
`cnn_calibrator.joblib`, the `per_mission` block the gate reads, and a new
`run_config` block recording folds, seed, **the augmentation magnitudes and the
view shapes**. Neither of those two was recoverable from run 1's summary, and
they are exactly what made its comparison against the incumbent unlike-for-like.

Augmentation is now declared in `conf/model/cnn_branches.yaml` at the same
magnitudes `conf/preprocess/default.yaml` gives the incumbent, rather than
being an implicit dataclass default.

### `compare_runs.py` — the reading, as a script rather than a session

Reading run 2 needs the same three measurements that were done ad hoc for run 1,
so they now live in `pipeline/scripts/compare_runs.py`: per-mission coverage
first, per-mission metrics with TESS marked as the gate and the aggregate marked
as never gating, then the AUC gap by quartile of transits actually caught.

Validated against run 1 — every recorded number reproduces from the artefacts:

```
Kepler     2238   0.9914  0.9566  +0.0348    R@1% 0.799 / 0.383
TESS       2367   0.9100  0.9079  +0.0021    R@1% 0.307 / 0.238   <- gates
all        4605   0.9558  0.9337  +0.0222                          <- never gates

Kepler gap by transits caught   Q1 +0.0245  Q2 +0.0335  Q3 +0.0416  Q4 +0.0944
TESS   gap on the same split    Q1 +0.0045  Q2 +0.0007  Q3 -0.0008  Q4 +0.0055
trend per quartile              Kepler +0.0218   TESS +0.0002
```

The TESS row is what makes the resolution reading more than a story: the same
quartile split on the mission we serve is flat to 0.0002 per step. And the
Kepler Q4 detail is the whole finding in one line — at a median of **1,035
transits caught**, the incumbent holds 0.9912 while the 301-bin branch model
falls to **0.8967**.

### The phase shift was degenerate at 301/31

`time_shift_frac` is a fraction of the bin count and the shift truncates to an
integer, so at the default 0.005:

```
global 2001  ->  +-10 bins      global 301  ->  +-1 bin
local  201   ->  +-1 bin        local   31  ->   0 bins
```

At 301/31 the coherent phase shift would have been **exactly zero on all six
local views** — a quarter of the augmentation silently absent on more than half
the branches. At the restored resolution it does what the incumbent's
augmentation does, which is the point of matching magnitudes rather than
copying the number. Noted in `viewset_augment.py` where the truncation happens.

This is not a bug in run 2 — it is a reason run 1 would have been a poor test of
augmentation even if it had had any.

### The candidate view set is now stale — flagged, not rebuilt

`data/processed/candidates_viewset/` holds **5,347 candidates at 301/31**
(built 2026-08-05). The resolution change does not touch it, so any branch model
from run 2 onward **cannot score candidates** until it is rebuilt at 2001/201 —
the input shapes simply will not match.

Not needed for run 2's promote/reject decision, which is measured entirely on
the labelled CV set. It *is* needed for:

- the observation-bias measurement on the candidate population
  (`candidate_bias.py`), where the original +0.211 finding lives;
- stage 2(b)'s control-arm host-pass rate;
- any shortlist produced by a promoted branch model.

Rebuilding is the same command with `--include-candidates`, at roughly the same
~60 targets/min, so budget about two hours. Deliberately deferred: it is wasted
work if run 2 falsifies the resolution hypothesis and the branch line is
reconsidered.

The three DVC-tracked artefacts the rebuild replaces —
`viewset.npz`, `viewset_scalars.parquet`, `viewset_tfrecords/` — already have
pointers, so they need `dvc add` (mine) then `dvc push` (Ollie's).

### "TESS is flat" was the wrong lesson — the deficit tracks transits, not mission

The within-mission quartile split showed Kepler's gap climbing +0.0245 ->
+0.0944 while TESS held at +0.0002 per step, which reads as "Kepler suffers,
TESS is immune". It is not. **Quartiles are cut per mission**, so TESS's top
quartile is a median of 89 transits caught where Kepler's is 1,035 — TESS looks
flat mostly because it never reaches the regime where the effect lives.

Cut both missions on the same absolute bands:

```
band        mission     n     inc       br       gap
0-10        Kepler    101   0.9615   0.9346   +0.0269
0-10        TESS      557   0.8745   0.8636   +0.0109
10-30       Kepler    206   0.9834   0.9581   +0.0253
10-30       TESS      873   0.8972   0.9063   -0.0091
30-100      Kepler    532   0.9895   0.9652   +0.0243
30-100      TESS      680   0.9356   0.9322   +0.0034
100-300     Kepler    629   0.9943   0.9553   +0.0390
100-300     TESS      195   0.9428   0.9404   +0.0024
300+        Kepler    770   0.9920   0.9229   +0.0690
300+        TESS       62   0.9243   0.8377   +0.0866
```

**Where TESS reaches 300+ transits it shows the largest gap in the table.**
n=62, so the point estimate is loose — but the direction is unambiguous and it
is the opposite of mission-immunity.

Two consequences. The mechanism claim is *stronger* than first written: a
resolution deficit should depend on how much real structure is folded into a
view, which is transit count, and it does — on both missions. And run 2 gets a
**second pre-registered test on the deployment mission**: the TESS 300+ band
must improve, not just Kepler.

Both cuts are now in `compare_runs.py`, so run 2 is read the same way rather
than re-derived.

The claim in the roadmap has been corrected. Note what was and was not wrong:
every number was right, and "TESS is flat on this split" was true as stated. The
error was the implication — that flatness meant TESS was unaffected, when it
meant TESS was mostly absent from the range. A per-group quantile cut compares
different populations under the same label.

### The mechanism, measured properly: an interaction, not one variable

The transit-count finding above raised an obvious sharper hypothesis — that what
matters is how many *bins the transit itself spans* on the coarse grid,
`duration / period x 301`. Measured on Kepler, that alone points the **wrong
way**: the gap grows with span (+0.0279 at <1 bin, +0.0624 at 8+). A pure
"transit too narrow to resolve" story is not what happened.

Span and count correlate (Spearman 0.44), so crossing them separates the two:

```
Kepler        span   transits     n   med span     inc       br       gap
               0-4      0-100   690        1.8  0.9835   0.9631   +0.0204
               0-4      100-+   149        3.1  0.9852   0.8406   +0.1446
               4-8      0-100   117        5.0  0.9952   0.9639   +0.0313
               4-8      100-+   413        5.9  0.9942   0.9399   +0.0543
               8-+      100-+   837       18.4  0.9915   0.9317   +0.0597

TESS           0-4      0-100   417        2.6  0.8899   0.8656   +0.0243
               4-8      0-100   576        5.9  0.8940   0.9026   -0.0086
               8-+      0-100  1117       13.0  0.9129   0.9188   -0.0059
               8-+      100-+   207       18.7  0.9426   0.9001   +0.0425
```

Within every span band the gap still tracks transit count, and it does so on
both missions independently — every low-count TESS cell sits at or below zero,
its one high-count cell at +0.0425.

**The worst cell by a factor of three is narrow-in-phase AND caught many
times**: 149 Kepler targets, median transit spanning **3.1 of 301 bins**, median
**134 transits caught**, incumbent 0.9852 against 0.8406. At 2001 bins those
transits span about 21 bins.

That is a mechanism rather than a story: many folded transits make the per-bin
median precise enough that real fine structure exists, and a coarse grid then
destroys it — worst where the feature is narrowest in phase. Both cuts are in
`compare_runs.py`.

**Pre-registered before the result:** if resolution is the cause, that +0.1446
cell improves most. A flat improvement across cells would mean the effect is
something else, whatever the headline Kepler number does.

## Run 2 built — sizing predictions held, and a benign race explained (2026-08-07)

**Serving still `ca906040` — untouched. Nothing promoted.** The 2001/201 view
set is built and sharded; CV is running.

### The pre-launch predictions, checked against the artefacts

| | predicted | measured |
|---|---:|---:|
| shard bytes | ~669 MB | **671 MB** |
| parameters | 233,617 | 233,617 |
| peak training RSS | ~5.4 GB | (CV in flight) |

The shard figure was extrapolated from a 296-target probe at 126.4 KB/example
and came in **0.3% off**. `viewset.npz` is 295 MB against 65 MB at 301/31.

Both shard assertions passed: **15 scalars** (13 was the merge-collision bug)
and `global_view [2001, 3]` / `local_view [201, 3]`.

### 5,426 examples, not 5,423 — a build that read a cache still being filled

Three more rows than run 1, and no rows lost. All three are TESS positives with
`lc_source == "ffi"`:

```
443607377  TESS  label 1  ffi  P=4.02 d   28 transits
335661164  TESS  label 1  ffi  P=0.84 d  150 transits
468350765  TESS  label 1  ffi  P=3.89 d   30 transits
```

The labels catalogue is unchanged (5,703 rows, untouched since 2026-08-01), and
`missing_fits` fell 273 -> 270 while `missing_ephemeris` held at 7. The cause is
timing: **run 1's view-set build finished at 00:20 on 2026-08-05, and the FFI
fetch did not finish writing `data/raw_ffi/` until 01:42** — 82 minutes later.
Run 1 read a cache another job was still filling and saw 4 FFI light curves
where 7 were coming.

This is the same class as the 2026-08-01 refresh-during-DV-pull race: a build
that reads a cache being written elsewhere produces a *silently smaller* data of
record, with no error anywhere. It cost 3 rows out of 5,423 (0.06%), so nothing
in run 1's conclusions moves — but the mechanism is worth naming, because
nothing in the pipeline would have reported it at any size.

**Consequence for the comparison:** run 2 trains on 3 rows run 1 did not have.
Immaterial to the AUC comparison, and the incumbent comparison runs on shared
rows regardless — but recorded so the population difference is not rediscovered
as a surprise.

## Run 2 — the resolution hypothesis is FALSIFIED (2026-08-07)

**Serving still `ca906040` — untouched. Nothing promoted; the registry is
unchanged.** Artefacts in `models/cv/branches-20260807-2001/`, with per-fold
checkpoints and calibration bundles this time.

### The result

`promotion_gate` → **REJECT**. Restoring 2001/201 made every slice worse and
roughly **doubled** the gap it was meant to close.

| slice | incumbent | run 1 (301/31) | run 2 (2001/201) | run 2 gap |
|---|---:|---:|---:|---:|
| TESS *(gates)* | 0.9100 | 0.9079 | **0.8944** | **+0.0156** |
| Kepler | 0.9914 | 0.9566 | **0.9207** | **+0.0707** |
| all | 0.9558 | 0.9337 | **0.9043** | **+0.0516** |
| TESS recall @1% FPR | 0.307 | 0.238 | **0.126** | |

Full slices (not just shared rows): TESS 0.8941, Kepler 0.9215, K2 0.8741,
all 0.9002 ± 0.0128.

The pre-registered rule was "Kepler gap stays above ~0.020 -> FALSIFIED". It
went from +0.0348 to **+0.0707**. The finer cell test fails the same way and
more informatively: **Kepler 0–10 transits went from +0.0269 to +0.2038**, so the
damage concentrates where evidence is *thinnest* — the opposite of what a
resolution deficit predicts.

**Under the Task 2(a) trigger the capacity run is now MANDATORY**, and per the
pre-registration this stops here. No tuning.

### The mechanism was a real correlation read causally, and that reading was wrong

Run 1's Kepler gap genuinely rose with transits caught, genuinely reached
+0.1446 in the narrow-transit / many-transit cell, and genuinely reproduced on
TESS wherever TESS reached that regime. Every one of those measurements survives.
What does not survive is the inference that 301 bins *caused* it: given more
bins, the model got worse everywhere, and worst on the sparsest data.

Worth keeping as a lesson about this specific trap: the covariate that predicted
the gap best (transits caught) is also a proxy for how much evidence a target
has, and "the model is weaker where evidence is thin" explains the same pattern
without any reference to bin counts. Run 2 is what distinguished them, and it
could not have been distinguished by more analysis of run 1.

### Three hypotheses tested and discarded before accepting the result

The result looked wrong (folds ran *faster* on 5.5x the data), so it was
checked before being believed rather than after.

1. **MC-dropout noise.** `cnn_branches.py` builds head dropout with
   `training=True` where the dual-view model uses `training=None`, and the
   comment claims it matches the dual-view model — it does not, and
   `mc_dropout_predict`'s own docstring specifies `training=None` as the
   contract. **But scoring is deterministic anyway**: six scorings of the same
   checkpoint gave 0.8927 with range **0.0000**. The `training=True` is still
   wrong as documentation and worth fixing, but it is not affecting any number.
2. **The checkpoint reload I added this session.** In-memory
   `restore_best_weights` and the reloaded `ModelCheckpoint` file were scored in
   one process on the same fold: **0.8991 and 0.8991**. The reload is sound.
3. **Undertraining.** Real but not the explanation — see the noise floor below.

### The noise floor, measured for the first time

Fold 0 re-run five times through `run_fold` with the trainer's own seeding, one
process each:

```
0.8927   0.8942   0.8984   0.9083   0.9179
mean 0.9023    sd 0.0106    range 0.0252
```

`set_global_seed`'s docstring already says it "doesn't make TF fully
deterministic on GPU", and nothing sets `enable_op_determinism`, so this is
known behaviour that had never been quantified. **Single-fold training sd is
≈ 0.011.**

**It does not overturn the result.** The 5-fold mean averages it down to between
0.0048 (folds independent) and 0.0106 (fully correlated); run 1 and run 2 differ
by 0.0313, which is **2.9σ to 6.6σ**. Note also that run 2's fold 0 (0.8927) is
the *lowest* of the five samples, so if anything run 2's headline is slightly
unlucky — and it is still 0.03 below run 1.

**But it does change how any `±` in this project should be read.** The std in
every `cv_summary.json` is the spread *across folds within one run*. It mixes
genuine fold-to-fold variation with training noise and says nothing about
whether re-running the same configuration reproduces the headline. **Any future
decision on a margin under ~0.02 needs repeat runs, not one run's fold std.**

An honest note on how this was reported: mid-investigation the spread was quoted
as 0.034 and flagged as possibly swamping the effect. That number included two
diagnostics that never called `set_global_seed`. Measured properly through the
trainer it is sd 0.0106, and the effect is comfortably outside it. The alarm was
overstated; the underlying gap in the project's uncertainty accounting was not.

### The one code defect the investigation left behind, now fixed

`cnn_branches.py` built head dropout with `training=True`, overriding the
call-time flag, with a comment claiming it matched the dual-view model. The
dual-view model uses `training=None`, and `mc_dropout_predict` documents
`training=None` as the contract it requires — so the setting was wrong on its
own terms and bought nothing.

It turned out **not** to affect any recorded number (scoring measured
deterministic to range 0.0000 over six draws), which is why it survived. Fixed
to `training=None` and pinned by
`test_scoring_is_deterministic_but_mc_dropout_still_works`, which asserts both
halves: `training=False` reproduces exactly, `training=True` varies.

Nothing needs re-running because of it — but a model built from the old code
scores stochastically under any future call that trusts the flag, which is
exactly what stage 4's serving parity will do.
