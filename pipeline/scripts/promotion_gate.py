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
    VERDICT_EXIT_CODES,
    evaluate_promotion,
    load_incumbent_summary,
    promote,
    write_decision,
)

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cv_summary", type=Path, help="cv_summary.json of the candidate run")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--brier-tolerance", type=float, default=0.005)
    parser.add_argument("--ece-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--incumbent-summary",
        type=Path,
        default=None,
        help=(
            "read the incumbent's metrics from this cv_summary.json instead of the one "
            "the registry names. Required whenever the served model's own summary predates "
            "the per_mission block — as ca906040's does, which makes the gate refuse every "
            "candidate on paperwork before comparing a single metric. Point it at "
            "models/cv/incumbent-rebaselined/cv_summary.json. The registry is not modified"
        ),
    )
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=None,
        help=(
            "override the shortlist-recall tolerance. Defaults to the candidate run's own "
            "measured floor (2 x seed_sd / sqrt(n_models_per_fold)); a run with no variance "
            "block falls back to the pre-stage-6 constant of 0.02"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "treat an unacknowledged alarm as blocking rather than advisory, giving "
            "UNRESOLVED instead of PROMOTE. An alarm owes a written explanation before "
            "promotion, which nobody is present to give on an unattended run — the "
            "weekly flow passes this. Alarms carrying a standing decision are listed in "
            "ACKNOWLEDGED_ALARMS and do not block"
        ),
    )
    parser.add_argument("--promote", action="store_true", help="Update the registry on success")
    parser.add_argument(
        "--verdict-out",
        type=Path,
        default=None,
        help=(
            "write the decision here as JSON. A caller that reads only the exit code cannot "
            "tell REJECT from a gate that crashed before deciding — both leave a non-zero "
            "status. This file is written only once a verdict exists, so its absence is the "
            "crash. The unattended refresh flow passes it and reports what it finds"
        ),
    )
    args = parser.parse_args()

    candidate = json.loads(args.cv_summary.read_text())
    incumbent = load_incumbent_summary(args.models_dir, args.incumbent_summary)

    decision = evaluate_promotion(
        candidate,
        incumbent,
        brier_tolerance=args.brier_tolerance,
        ece_tolerance=args.ece_tolerance,
        recall_tolerance=args.recall_tolerance,
        strict=args.strict,
    )
    log.info("[promotion] %s", decision)

    # Recorded the moment it exists, before the registry is touched. A promotion
    # that is decided and then fails to apply is a different event from a
    # rejection, and writing this afterwards would lose exactly that case.
    if args.verdict_out is not None:
        write_decision(args.verdict_out, decision)

    if decision.promoted and args.promote:
        run_id = args.cv_summary.parent.name
        registry = promote(args.models_dir, run_id, args.cv_summary)
        log.info("[promotion] registry updated -> run %s", registry["run_id"])

    # Three exit codes for three verdicts. UNRESOLVED must not collapse into
    # either neighbour: read as 0 an unattended loop would promote on a margin
    # it cannot resolve, and read as 1 it would report a quality rejection that
    # did not happen.
    sys.exit(VERDICT_EXIT_CODES[decision.verdict])


if __name__ == "__main__":
    main()
