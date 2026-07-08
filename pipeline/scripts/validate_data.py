"""Run the data validation gates against the on-disk artefacts.

Usage (from the repository root):

    python pipeline/scripts/validate_data.py                  # validate whatever exists
    python pipeline/scripts/validate_data.py --strict         # missing artefact = failure
    python pipeline/scripts/validate_data.py \
        --previous-labels path/to/old/labels.parquet          # + refresh leakage guard

Exit code 0 = every gate passed; 1 = at least one failed. Designed to slot
directly into the refresh DAG (orchestrator branch) and pre-training checks.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pandera.errors

from exoplanet_hunter.datasets import load_views
from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation import (
    assert_refresh_safe,
    candidate_catalogue_schema,
    check_views,
    label_catalogue_schema,
)

log = get_logger(__name__)


def _gate(name: str, fn: Callable[[], object]) -> bool:
    try:
        fn()
        log.info("[gate] %-22s PASS", name)
        return True
    except pandera.errors.SchemaErrors as exc:
        log.error("[gate] %-22s FAIL\n%s", name, exc.failure_cases.head(20))
        return False
    except Exception as exc:
        log.error("[gate] %-22s FAIL: %s", name, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/labels.parquet"))
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/catalogue/candidates.parquet")
    )
    parser.add_argument("--views", type=Path, default=Path("data/processed/views.npz"))
    parser.add_argument(
        "--previous-labels",
        type=Path,
        default=None,
        help="Previous labels.parquet — enables the refresh leakage guard",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat missing artefacts as failures instead of skips"
    )
    args = parser.parse_args()

    results: list[bool] = []

    for name, path, fn in (
        (
            "label-catalogue",
            args.labels,
            lambda: label_catalogue_schema.validate(pd.read_parquet(args.labels), lazy=True),
        ),
        (
            "candidate-catalogue",
            args.candidates,
            lambda: candidate_catalogue_schema.validate(
                pd.read_parquet(args.candidates), lazy=True
            ),
        ),
    ):
        if path.exists():
            results.append(_gate(name, fn))
        elif args.strict:
            log.error("[gate] %-22s FAIL: %s missing", name, path)
            results.append(False)
        else:
            log.info("[gate] %-22s SKIP (%s not built yet)", name, path)

    if args.views.exists():

        def _views_gate() -> None:
            problems = check_views(load_views(args.views))
            if problems:
                raise ValueError("; ".join(problems))

        results.append(_gate("views", _views_gate))
    elif args.strict:
        log.error("[gate] %-22s FAIL: %s missing", "views", args.views)
        results.append(False)
    else:
        log.info("[gate] %-22s SKIP (%s not built yet)", "views", args.views)

    if args.previous_labels is not None:

        def _leakage_gate() -> None:
            flips = assert_refresh_safe(
                pd.read_parquet(args.previous_labels),
                pd.read_parquet(args.labels),
            )
            if len(flips):
                log.warning(
                    "[gate] %d label flips quarantined (since-confirmed holdout):\n%s",
                    len(flips),
                    flips.to_string(index=False),
                )

        results.append(_gate("refresh-leakage", _leakage_gate))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
