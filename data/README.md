# Data directory — fresh artefacts only

Nothing in this directory is committed to git, and **no preprocessed
artefacts are ported from V1**. Every catalogue, light-curve cache, and
views/TFRecord set here is regenerated from NASA sources by the V2 pipeline
(`pipeline/scripts/build_dataset.py` / `preprocess_only.py`), so the tf.data
input pipeline and the validation gates are exercised against data they
produced themselves.

Once `feat/dvc-versioning` lands, the contents are tracked by DVC with R2 as
the remote: git holds the pointers, R2 holds the bytes.

## Current layout

```
raw/exofop/tois.csv    ExoFOP TOI bulk export   (exofop.ipac.caltech.edu/tess/download_toi.php?output=csv)
raw/exofop/ctois.csv   ExoFOP CTOI bulk export  (exofop.ipac.caltech.edu/tess/download_ctoi.php?output=csv)
catalogue/candidates.parquet   normalised TOI+CTOI candidate catalogue (API reads this)
catalogue/candidates.csv       same table as a portable CSV export
```

Rebuild the catalogue after refreshing the raw exports:

    python pipeline/scripts/ingest_exofop.py
