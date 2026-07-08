# Data directory — fresh artefacts only

Nothing in this directory is committed to git, and **no preprocessed
artefacts are ported from V1**. Every catalogue, light-curve cache, and
views/TFRecord set here is regenerated from NASA sources by the V2 pipeline
(`pipeline/scripts/build_dataset.py` / `preprocess_only.py`), so the tf.data
input pipeline and the validation gates are exercised against data they
produced themselves.

Once `feat/dvc-versioning` lands, the contents are tracked by DVC with R2 as
the remote: git holds the pointers, R2 holds the bytes.
