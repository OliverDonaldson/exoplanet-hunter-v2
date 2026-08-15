# Orchestration

`flows/refresh_pipeline.py` is the V2 DAG on Prefect 3:

    ExoFOP exports → candidate catalogue → label catalogue (TAP) →
    validation gates (+ leakage guard vs previous labels) →
    [dataset changed materially?] → preprocess → shard → train →
    promotion gate → DVC publish to R2

## Running

```
make refresh                                            # full refresh, trains if warranted
python orchestration/flows/refresh_pipeline.py --no-train      # refresh + gates only

# Expansion run: full TESS pool + Kepler + K2, 13-dim aux. The Kepler FITS
# cache lives in data/raw/kepler/lightcurves; KEPLER_RAW_DIR only overrides that location.
python orchestration/flows/refresh_pipeline.py --force-train --data-config full
```

No Prefect server required — flows run standalone with an ephemeral local
API. For the run-history UI: `prefect server start`, then run flows in
another terminal. Scheduling is a one-liner once a work pool exists
(`refresh_pipeline.serve(cron="0 6 * * 1")` weekly, e.g.).

## The two decisions that cost money

* **"Changed materially"** (fires training): defined in
  `exoplanet_hunter.validation.trigger` — ≥ `min_new_labelled` genuinely
  new labelled targets (confirmed + false positives), or an explicit
  `--force-train` expansion run. Label flips never count: the leakage
  guard quarantines them into the since-confirmed holdout.
* **Where training runs**: locally by default; set `BURST_CMD` to dispatch
  the same step to a rented GPU (provision → run the train container
  against R2-synced shards → tear down). The flow doesn't care which.
