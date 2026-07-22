# System architecture — orchestration, pipeline, model, serving

End-to-end map of Exoplanet Hunter V2: from the catalogues and mission
archives, through the self-refreshing training pipeline and promotion gate,
to the live scoring API and vetting console. The training path is a genuine
sequence (ingest → preprocess → train → gate → serve); the Prefect flow wraps
it and the weekly trigger closes the loop.

```mermaid
flowchart TB
  subgraph SRC["1 · External data sources"]
    direction LR
    NEA["NASA Exoplanet Archive · TAP<br/>ps · toi · cumulative KOI<br/>(label source of truth)"]
    EXOFOP["ExoFOP-TESS<br/>TOI + CTOI · transit SNR<br/>(enrichment)"]
    MAST["MAST<br/>TESS SPOC 2-min light curves"]
    STSCI["STScI archive<br/>Kepler long-cadence LCs"]
    TICS["MAST Catalogs<br/>TIC / stellar params"]
  end

  subgraph ORCH["2 · Orchestration — Prefect flow (refresh_pipeline.py)"]
    direction LR
    PLIST["launchd plist · Sat 09:00<br/>or manual --force-train"]
    TRIG["validation/trigger.py<br/>retrain decision (data delta)"]
  end

  subgraph INGEST["3 · Ingestion (exoplanet_hunter/data)"]
    direction LR
    CAT["catalog.py"]
    EXO["exofop.py"]
    DL["download.py<br/>fetch + manifest lock"]
    STEL["stellar.py"]
  end

  subgraph STORE["4 · Data layer / storage"]
    direction LR
    LABELS[("labels.parquet")]
    CANDS[("candidates.parquet")]
    RAW[("data/raw* + manifest.json<br/>FITS cache · ~81 GB")]
    DVC[("DVC → Cloudflare R2<br/>data + model artefacts")]
    MLF[("MLflow · sqlite")]
  end

  subgraph PIPE["5 · Preprocess → dataset (build_dataset.py · preprocess · datasets)"]
    direction LR
    CLEAN["clean<br/>σ-clip"]
    FLAT["flatten<br/>transit-masked Savitzky–Golay"]
    VIEWS["fold → global + local views<br/>+ 13-dim vetting aux"]
    SHARD["shard → TFRecords<br/>tf.data + aux transform"]
  end

  subgraph TRAINING["6 · Training (training · models)"]
    direction LR
    CV["train.py · 5-fold CV<br/>dual-view CNN + MC-Dropout"]
    CAL["Platt calibration<br/>+ F1 threshold sweep"]
    TUNE["Optuna · tune.py"]
  end

  GATE{{"7 · Promotion gate (validation/promotion.py)<br/>beat CV AUC · Brier + ECE guard"}}
  REG[("models/registry.json<br/>+ cv/RUN/fold_* bundles")]

  subgraph SERVE["8 · Serving API — FastAPI on Fly.io (api · scoring)"]
    direction TB
    SCORE["GET /score/TIC · TargetScorer<br/>download → ephemeris (user → catalogue → BLS)<br/>→ clean/flatten → views → 5-fold ensemble + MC<br/>→ calibrate → diagnostics → verdict"]
    DIAG["diagnostics.py cautions<br/>centroid · odd/even depth + timing<br/>secondary §3.9 · duration §3.4 · FA bundle"]
    EP["GET /healthz · /reliability · /candidates"]
  end

  subgraph UICONSOLE["9 · Console — React on Render (frontend)"]
    direction LR
    TABLE["candidate table"]
    PANEL["vetting panel<br/>phase views · probability bar<br/>caution chips · reliability diagram"]
  end

  WEBHOOK["NOTIFY_WEBHOOK_URL<br/>promotion verdict"]

  PLIST --> TRIG --> CAT
  NEA --> CAT --> LABELS
  EXOFOP --> EXO --> CANDS
  MAST --> DL
  STSCI --> DL --> RAW
  TICS --> STEL
  LABELS --> CLEAN
  RAW --> CLEAN --> FLAT --> VIEWS --> SHARD
  STEL --> VIEWS
  SHARD --> CV --> CAL --> GATE
  TUNE -.-> CV
  CV --> MLF
  GATE -->|promoted| REG
  GATE --> DVC
  REG --> SCORE
  CANDS --> SCORE
  RAW --> SCORE
  SCORE --> DIAG
  REG --> EP
  SCORE --> PANEL
  CANDS --> TABLE
  EP --> PANEL
  GATE -->|verdict| WEBHOOK

  classDef src fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;
  classDef store fill:#fef3c7,stroke:#d97706,color:#78350f;
  classDef compute fill:#dcfce7,stroke:#16a34a,color:#14532d;
  classDef serve fill:#f3e8ff,stroke:#9333ea,color:#581c87;
  class NEA,EXOFOP,MAST,STSCI,TICS src;
  class LABELS,CANDS,RAW,DVC,MLF,REG store;
  class CLEAN,FLAT,VIEWS,SHARD,CV,CAL,TUNE compute;
  class SCORE,DIAG,EP,TABLE,PANEL serve;
```

**Legend** — blue: external sources · amber: data/storage · green:
compute (preprocess + train) · purple: serving (API + console).

**Reading it.** The weekly `launchd` plist (or a manual `--force-train`) kicks
the Prefect flow, which asks `trigger.py` whether the catalogue changed enough
to retrain. Ingestion pulls labels from the NASA Exoplanet Archive, candidate
metadata + SNR from ExoFOP, and light curves from MAST/STScI into the parquet
+ FITS cache. `build_dataset.py` cleans, flattens (masking the transit),
folds, and builds the two phase-views plus the 13-dim vetting-aux vector, then
shards to TFRecords. `train.py` fits the 5-fold dual-view CNN with MC-Dropout
and Platt calibration; the promotion gate only advances a run that beats the
incumbent on CV AUC without degrading Brier/ECE, updating `registry.json`.
Artefacts version to R2 via DVC. The Fly API loads the registered ensemble and
scores any TIC on demand — resolving the ephemeris, rebuilding the same views,
running the ensemble, then layering the LEO-Vetter cautions — and the Render
console renders it. A promotion posts its verdict to the notify webhook.
