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
3. **Build the validation runners against the served model.** (a) the
   injection-recovery runner on eval/injection_recovery (drive real LCs through
   preprocess + 13-dim aux + ensemble → completeness curve — the defensible
   sensitivity number CV-AUC can't give); (b) an FPP/NFPP shortlist run
   (`validate_candidates.py --insecure-trilegal`). Optional review gaps here:
   ephemeris-match test, statistical-bootstrap FA (DV §3.5).
4. **Step 3 — since-confirmed holdout eval.** Data-gated: needs a few weekly
   Saturday refreshes to accumulate flips, then run eval_since_confirmed.py.
5. **Step 4 — tidy sweep.** score_candidates.py still builds legacy 9-dim aux
   (rework to share the 13-dim path before the next shortlist run); stale
   docstrings.
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
