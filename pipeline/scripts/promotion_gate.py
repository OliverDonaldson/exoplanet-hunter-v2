"""Decide whether a fresh CV run replaces the incumbent model.

Usage (from the repository root):

    # dry run — print the decision, exit 0 iff it would promote
    python pipeline/scripts/promotion_gate.py models/cv/<run_id>/cv_summary.json

    # apply — update models/registry.json on success
    python pipeline/scripts/promotion_gate.py models/cv/<run_id>/cv_summary.json --promote
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation import (
    evaluate_promotion,
    load_incumbent_summary,
    promote,
)

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cv_summary", type=Path, help="cv_summary.json of the candidate run")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--brier-tolerance", type=float, default=0.005)
    parser.add_argument("--ece-tolerance", type=float, default=0.01)
    parser.add_argument("--promote", action="store_true", help="Update the registry on success")
    args = parser.parse_args()

    candidate = json.loads(args.cv_summary.read_text())
    incumbent = load_incumbent_summary(args.models_dir)

    decision = evaluate_promotion(
        candidate,
        incumbent,
        brier_tolerance=args.brier_tolerance,
        ece_tolerance=args.ece_tolerance,
    )
    log.info("[promotion] %s", decision)

    if decision.promoted and args.promote:
        run_id = args.cv_summary.parent.name
        registry = promote(args.models_dir, run_id, args.cv_summary)
        log.info("[promotion] registry updated -> run %s", registry["run_id"])

    sys.exit(0 if decision.promoted else 1)


if __name__ == "__main__":
    main()
