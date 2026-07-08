"""Build the candidate catalogue from ExoFOP TOI + CTOI exports.

Writes both a parquet (read by the API / future DuckDB queries) and a plain
CSV (the browsable/downloadable "all of our data" artefact):

    data/catalogue/candidates.parquet
    data/catalogue/candidates.csv

Usage (from the repository root):

    python pipeline/scripts/ingest_exofop.py
    python pipeline/scripts/ingest_exofop.py --toi path/to/tois.csv --out-dir data/catalogue
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exoplanet_hunter.data.exofop import build_candidate_catalogue
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toi", type=Path, default=Path("data/raw/exofop/tois.csv"))
    parser.add_argument("--ctoi", type=Path, default=Path("data/raw/exofop/ctois.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/catalogue"))
    args = parser.parse_args()

    catalogue = build_candidate_catalogue(args.toi, args.ctoi)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "candidates.parquet"
    csv_path = args.out_dir / "candidates.csv"
    catalogue.to_parquet(parquet_path, index=False)
    catalogue.to_csv(csv_path, index=False)

    log.info("[ingest] wrote %s and %s", parquet_path, csv_path)
    log.info("[ingest] by source:\n%s", catalogue["source"].value_counts().to_string())
    log.info(
        "[ingest] by TFOPWG disposition:\n%s",
        catalogue["disposition"].fillna("(none)").value_counts().to_string(),
    )


if __name__ == "__main__":
    main()
