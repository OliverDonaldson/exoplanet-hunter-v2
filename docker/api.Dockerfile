# Serving image: FastAPI + the pipeline package (exact training-time preprocessing).
# Build from the repository root:
#   docker build -f docker/api.Dockerfile -t exoplanet-hunter-api .
FROM python:3.11-slim

WORKDIR /srv

# Install the pipeline first (heavy scientific deps cache as their own layer),
# then the thin API layer on top.
COPY pipeline/ pipeline/
RUN pip install --no-cache-dir ./pipeline

COPY api/ api/
RUN pip install --no-cache-dir ./api

# Model + calibration bundles are mounted (or pulled from R2 at startup),
# never baked into the image.
ENV MODEL_DIR=/srv/models
VOLUME /srv/models

WORKDIR /srv/api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
