# exoplanet-hunter (pipeline package)

The ML pipeline for Exoplanet Hunter V2: catalogue ingest, light-curve
preprocessing (clean / flatten / fold / views), the dual-view 1D CNN with
MC-Dropout and Platt calibration, BLS search, vetting diagnostics, and
evaluation. The FastAPI serving layer (`../api`) installs this package so
inference always runs the exact training-time preprocessing.

Calibration is Platt scaling, not temperature scaling — temperature has no bias
term and so cannot correct the distribution shift that once produced a 0.136
ECE regression.

## Layout

| path | what lives there |
|---|---|
| `src/exoplanet_hunter/data/` | catalogue TAP builders, downloader, stellar params |
| `src/exoplanet_hunter/preprocess/` | clean → flatten → fold → views |
| `src/exoplanet_hunter/features/` | aux vector, pink noise, centroid, follow-up observables |
| `src/exoplanet_hunter/datasets/` | views ↔ npz ↔ tf.data, aux normalisation |
| `src/exoplanet_hunter/models/` | the dual-view CNN |
| `src/exoplanet_hunter/training/` | CV trainer, calibration, Optuna tuning |
| `src/exoplanet_hunter/scoring/` | `TargetScorer` — the serving path |
| `src/exoplanet_hunter/search/` | BLS period search |
| `src/exoplanet_hunter/eval/` | metrics, vetting figures, injection-recovery |
| `src/exoplanet_hunter/validation/` | Pandera schemas, gates, promotion, TRICERATOPS |
| `scripts/` | Hydra entry points — see [scripts/README.md](scripts/README.md) |
| `conf/` | Hydra config groups (`data`, `preprocess`, `model`, `train`) |
| `vendor/` | patched third-party source — see [vendor/triceratops](vendor/triceratops/README.md) |
| `tests/` | the suite |

## Working on it

```bash
pytest pipeline/tests -m "not network and not slow"
pip install -e ./pipeline[dev]
```

For the shortlist validation layer, install the **vendored** TRICERATOPS, not
PyPI's — stock 1.0.20 satisfies the same constraint and returns different,
sometimes constant, FPP values:

```bash
pip install -e ./pipeline[validation] && pip install -e pipeline/vendor/triceratops --no-deps
```

`tests/test_vendor_triceratops.py` fails if the environment resolves a stock
install instead.

## One implementation per job

The aux vector lives only in `features/aux.py::build_aux_row`; the catalogue
request only in `data/catalog.py::build_labels_from_cfg`; preprocessing only in
`scripts/build_dataset.py`. These are not style preferences — each rule exists
because a second copy silently produced wrong data (9-dim aux into a 13-dim
build; a hardcoded request that dropped every K2 row).

See [docs/index.md](../docs/index.md) for the documentation map and
[docs/features.md](../docs/features.md) for what the model consumes.
