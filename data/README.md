# Data directory — fresh artefacts only

Nothing in this directory is committed to git, and **no preprocessed
artefacts are ported from V1**. Every catalogue, light-curve cache, and
views/TFRecord set here is regenerated from NASA sources by the V2 pipeline
(`pipeline/scripts/build_dataset.py` / `preprocess_only.py`), so the tf.data
input pipeline and the validation gates are exercised against data they
produced themselves.

Everything except the raw FITS caches is tracked by **DVC** with R2 as the
remote: git holds the `.dvc` pointers, R2 holds the bytes. `dvc pull`
materialises them on a fresh clone; `dvc push` syncs after a rebuild
(`make data-push` / `make data-pull`). The FITS caches (`raw/`,
`raw_kepler/`) stay local-only — NASA hosts the source of truth, so they
re-download on demand and can be evicted freely.

## Current layout

```
exofop/tois.csv        ExoFOP TOI bulk export   (exofop.ipac.caltech.edu/tess/download_toi.php?output=csv)
exofop/ctois.csv       ExoFOP CTOI bulk export  (exofop.ipac.caltech.edu/tess/download_ctoi.php?output=csv)
catalogue/             normalised TOI+CTOI candidate catalogue (parquet + CSV export)
labels/                training label catalogue from the NASA TAP queries
processed/views.npz    phase-folded global/local views + aux features
processed/tfrecords/   sharded TFRecord set streamed by the trainer
raw/                   TESS FITS cache — local-only, evictable
```

Rebuild the catalogue after refreshing the ExoFOP exports:

    python pipeline/scripts/ingest_exofop.py && dvc add data/exofop data/catalogue
