# Serving image: FastAPI + the pipeline package (exact training-time
# preprocessing). Model + catalogue artefacts are pulled from R2 by the
# entrypoint at boot, so the image stays artefact-free and the registry in
# git decides what gets served.
#
# Build from the repository root:
#   docker build -f docker/api.Dockerfile -t exoplanet-hunter-api .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Heavy scientific deps cache as their own layer; dvc[s3] fetches artefacts.
# Constraints pin training-time versions and stop pip's resolver backtracking.
# build-essential exists only within this layer: batman-package (via
# transitleastsquares) ships source-only; purged after the wheels are built.
COPY docker/constraints.txt docker/constraints.txt
COPY pipeline/ pipeline/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -c docker/constraints.txt ./pipeline "dvc[s3]>=3.30" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY api/ api/
RUN pip install --no-cache-dir -c docker/constraints.txt ./api

# DVC pointers + config and the model registry (all tiny, all in git).
COPY .dvc/config .dvc/config
COPY .dvcignore ./
COPY data/catalogue.dvc data/
COPY models/ models/
COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
RUN chmod +x /usr/local/bin/api-entrypoint.sh

# Artefacts land under the repo-shaped tree the code expects.
ENV MODEL_DIR=/srv/models \
    DATA_RAW_DIR=/srv/data/raw \
    CATALOGUE_PATH=/srv/data/catalogue/candidates.parquet

EXPOSE 8000
CMD ["api-entrypoint.sh"]
