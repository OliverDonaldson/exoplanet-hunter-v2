"""Run the data validation gates against the on-disk artefacts.

Usage (from the repository root):

    python pipeline/scripts/validate_data.py                  # validate whatever exists
    python pipeline/scripts/validate_data.py --strict         # missing artefact = failure
    python pipeline/scripts/validate_data.py \
        --previous-labels path/to/old/labels.parquet          # + leakage and shrink guards
    python pipeline/scripts/validate_data.py \
        --previous-labels ... --allow-shrink                  # intentional reduction

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
from exoplanet_hunter.datasets.viewset_io import ViewSetArrays
from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation import (
    assert_refresh_safe,
    candidate_catalogue_schema,
    check_catalogue_shrink,
    check_dv_archive,
    check_view_set,
    check_views,
    label_catalogue_schema,
    record_quarantine,
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
    parser.add_argument("--labels", type=Path, default=Path("data/tables/labels/labels.parquet"))
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/tables/catalogue/candidates.parquet")
    )
    parser.add_argument("--views", type=Path, default=Path("data/processed/views.npz"))
    parser.add_argument("--dv", type=Path, default=Path("data/raw/tess/dv"))
    parser.add_argument("--viewset", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--previous-labels",
        type=Path,
        default=None,
        help="Previous labels.parquet — enables the leakage and shrink guards",
    )
    parser.add_argument(
        "--max-shrink-frac",
        type=float,
        default=0.10,
        help="Fraction of the previous label catalogue that may disappear (default 0.10)",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="Accept a reduction the shrink guard would otherwise reject",
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

    if (args.viewset / "viewset.npz").exists():

        def _viewset_gate() -> None:
            problems = check_view_set(ViewSetArrays.load(args.viewset))
            if problems:
                raise ValueError("; ".join(problems))

        results.append(_gate("view-set", _viewset_gate))
    elif args.strict:
        log.error("[gate] %-22s FAIL: %s missing", "view-set", args.viewset / "viewset.npz")
        results.append(False)
    else:
        log.info("[gate] %-22s SKIP (%s not built yet)", "view-set", args.viewset)

    if args.dv.exists():

        def _dv_gate() -> None:
            # Expected set = the TESS targets the pipeline knows about, so an
            # interrupted fetch is caught as "never queried" rather than
            # silently masked out as "this target has no DV products".
            expected = None
            if args.labels.exists():
                labels = pd.read_parquet(args.labels)
                expected = labels[labels["mission"] == "TESS"]["tic_id"].astype(int).tolist()
            problems = check_dv_archive(args.dv, expected)
            if problems:
                raise ValueError("; ".join(problems))

        results.append(_gate("dv-archive", _dv_gate))
    elif args.strict:
        log.error("[gate] %-22s FAIL: %s missing", "dv-archive", args.dv)
        results.append(False)
    else:
        log.info("[gate] %-22s SKIP (%s not built yet)", "dv-archive", args.dv)

    if args.previous_labels is not None:

        def _shrink_gate() -> None:
            problems = check_catalogue_shrink(
                pd.read_parquet(args.previous_labels),
                pd.read_parquet(args.labels),
                max_shrink_frac=args.max_shrink_frac,
            )
            if not problems:
                return
            detail = "; ".join(problems)
            if args.allow_shrink:
                log.warning("[gate] %-22s shrink allowed: %s", "label-shrink", detail)
                return
            raise ValueError(f"{detail} — pass --allow-shrink if the reduction is intentional")

        results.append(_gate("label-shrink", _shrink_gate))

        def _leakage_gate() -> None:
            flips = assert_refresh_safe(
                pd.read_parquet(args.previous_labels),
                pd.read_parquet(args.labels),
            )
            if len(flips):
                # Recorded, not merely reported. This logged "quarantined" while
                # writing nothing and nothing applying it, so under the 2% flip
                # threshold every flipped row stayed eligible for training.
                quarantine = record_quarantine(flips, args.labels.parent)
                log.warning(
                    "[gate] %d label flips quarantined (%d held out of training in total):\n%s",
                    len(flips),
                    len(quarantine),
                    flips.to_string(index=False),
                )

        results.append(_gate("refresh-leakage", _leakage_gate))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
