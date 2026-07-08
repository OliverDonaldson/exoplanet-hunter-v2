# exoplanet-hunter (pipeline package)

The ML pipeline for Exoplanet Hunter V2: catalogue ingest, light-curve
preprocessing (clean / flatten / fold / views), the dual-view 1D CNN with
MC-Dropout and temperature calibration, BLS/TLS search, centroid vetting,
and evaluation. The FastAPI serving layer (`../api`) installs this package
so inference always runs the exact training-time preprocessing.

See the repository-root README for the full V2 architecture and build order.
