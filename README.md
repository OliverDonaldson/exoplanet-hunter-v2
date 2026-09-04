# Exoplanet Hunter V2

A calibrated deep-learning pipeline for **vetting** transit candidates in NASA
TESS, Kepler and K2 photometry. Given a known signal, a TOI or a KOI, it returns
the probability that the signal is a planet together with the diagnostic
evidence behind it, so that follow-up telescope time goes to the candidates
that deserve it. It classifies signals other pipelines detected; it does not
search for new ones.

Catalogue refresh → validation gates → phase-folded views → 5-fold CNN
ensemble → Platt calibration → promotion gate → FastAPI scoring → vetting
console. The governing rule is **beat the baseline before you cheer**: a model
ships only if it beats the champion's cross-validated ROC-AUC without losing
calibration or shortlist recall, and every margin is read against a noise floor, the
seed-to-seed spread of the same metric measured in the same run.

## What is served

Run `ca906040`, promoted 2026-07-19: a dual-view CNN over the global and local
phase-folded views plus nine scalars, five folds, MC-dropout for uncertainty,
Platt scaling for calibration. Out-of-fold, per mission:

| Mission | n | ROC-AUC | Recall @1% FPR | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| TESS (gates promotion) | 2,367 | 0.910 | 0.307 | 0.121 | 0.044 |
| Kepler (diagnostic) | 2,238 | 0.991 | 0.799 | 0.036 | 0.041 |

Out of fold, from `models/cv/champion-rebaselined-today/cv_summary.json`, the
champion re-scored on the current labelled set on 2026-08-17. The pooled figure
(0.956 ROC-AUC across both missions) is recorded but not read as a headline: it
averages two missions with different label provenance and class balance, and
TESS is the mission the service scores. K2 (530 labelled rows) is in the
catalogue, but the served run predates it and has no K2 slice; the promotion
gate flags that permanently. The deployed Model page still prints a noise floor
measured on a different architecture; that is issue #11.

Live: the console at https://exoplanet-hunter-console.onrender.com and the API
at https://exoplanet-hunter-api.fly.dev. Maintained by Oliver Donaldson.

## Layout

```
pipeline/       the science: ingest, preprocess, features, models, training, eval, validation
api/            FastAPI serving; the wire contract is app/schemas.py
frontend/       the vetting console (design-console/, one static file)
orchestration/  the weekly Prefect refresh: gates, train if warranted, promotion gate, publish
docker/         the API image (Fly) and the on-demand training image
infra/          R2 layout and secrets policy
data/           DVC-tracked artefacts; nothing here is committed
models/         registry.json; promoted runs are DVC-tracked
docs/           start at docs/index.md
```

## Quickstart

```bash
conda env create -f environment.yml && conda activate exoplanet-hunter-v2
pre-commit install --hook-type pre-commit --hook-type commit-msg
dvc pull          # artefacts from R2; needs credentials (infra/README.md)
make test         # the fast pipeline and API suites
make api          # FastAPI on :8000
make frontend     # build the console and serve it on :5173
```

Always activate `exoplanet-hunter-v2` first. The V1 environment carries V1's
code under the same package name and runs last year's pipeline without a word.
The test suites run without the DVC artefacts; scoring, the console and the
refresh need them pulled.

## Documentation

[docs/index.md](docs/index.md) maps every document. `docs/PLAN.md` is where the
project stands; `docs/experiments/` is the record of what was measured.

## Licence

MIT, see [LICENSE](LICENSE). `pipeline/vendor/triceratops` is a patched copy of
TRICERATOPS under its own licence, with the patches listed in its README.
ExoMiner is reimplemented and credited, never vendored (NASA NOSA licence).
