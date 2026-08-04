"""Observation bias on the CANDIDATE population, where the original finding lives.

The roadmap's +0.211 / -0.003 was measured over the 3,919 scored candidates, not
the labelled CV set, so it could not be compared with anything measured on
labels. This recomputes the incumbent's candidate-population correlations from
scratch, reporting both covariates so that the original figures can be
identified rather than assumed.

What it found (2026-08-05, 3,908 rows):

    baseline_days            +0.2075   <- the roadmap's +0.211, correct
    expected_transit_count   -0.0025   <- the roadmap's -0.003, mislabelled
    observed_transit_count   -0.0476   <- transits actually captured

So the baseline figure was sound and the transit-count figure was measured
against predicted rather than captured transits.

Candidates carry no labels (`label == -1`), so the label-structure comparison
made on the CV set cannot be repeated here. The reference is the labelled set's
own label correlation: +0.278 all-mission, +0.387 on TESS. Every scored
candidate is TESS, so +0.387 is the comparison that matters and +0.208 sits well
below it.

Usage (from the repository root):

    python pipeline/scripts/candidate_bias.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.eval.observation_bias import baseline_days, measure_observation_bias
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)


def _report(frame: pd.DataFrame, scores: np.ndarray, name: str) -> dict[str, float]:
    days = measure_observation_bias(scores, frame)
    count = measure_observation_bias(scores, frame, baseline_column="expected_transit_count")
    row = {
        "n": len(frame),
        "baseline_days": days.baseline_sensitivity,
        "baseline_expected_count": count.baseline_sensitivity,
        "transit": days.transit_sensitivity,
        "completeness": days.completeness_sensitivity,
    }
    log.info(
        "[candidate-bias] %-12s n=%4d  baseline(days) %+.3f  baseline(count) %+.3f  "
        "transit %+.3f  completeness %+.3f",
        name,
        row["n"],
        row["baseline_days"],
        row["baseline_expected_count"],
        row["transit"],
        row["completeness"],
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, default=Path("results/candidates_scored.parquet"))
    parser.add_argument(
        "--viewset",
        type=Path,
        default=Path("data/processed/candidates_viewset/viewset_scalars.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/candidate_observation_bias.json"))
    args = parser.parse_args()

    scored = pd.read_parquet(args.scored)
    views = pd.read_parquet(args.viewset)

    usable = scored[scored["prob_mean"].notna()]
    log.info(
        "[candidate-bias] scored %d rows, %d with a probability (%s)",
        len(scored),
        len(usable),
        scored["status"].value_counts().to_dict(),
    )

    # Suffix rather than trust: the labelled build lost two scalars to an
    # unsuffixed merge collision that passed all seven gates.
    joined = usable.merge(views, on="tic_id", how="inner", suffixes=("_scored", "_view"))
    if joined["tic_id"].duplicated().any():
        raise ValueError("tic_id is not unique after the join")
    log.info(
        "[candidate-bias] joined %d of %d scored candidates to a view set row",
        len(joined),
        len(usable),
    )

    # The join brings two periods. The covariate must use the one the transit
    # counts were derived from, or the reconstruction is of a different span.
    joined["period"] = joined["period_view"]
    disagree = ~np.isclose(joined["period_view"], joined["period_scored"], rtol=1e-3)
    log.info(
        "[candidate-bias] period disagreement between scored and view rows: %d (%.1f%%)",
        int(disagree.sum()),
        100.0 * disagree.mean(),
    )

    days = baseline_days(joined)
    log.info(
        "[candidate-bias] baseline_days: median %.1f  IQR %.1f-%.1f  max %.1f  zero for %d (%.1f%%)",
        float(np.median(days)),
        float(np.percentile(days, 25)),
        float(np.percentile(days, 75)),
        float(days.max()),
        int((days == 0).sum()),
        100.0 * float((days == 0).mean()),
    )

    # Independent check on the reconstruction: DV's own sector count is sourced
    # from the archive, not from our ephemeris arithmetic. TESS sectors are ~27 d,
    # so the two should rank together if the reconstruction means anything.
    has_sectors = joined["n_sectors_observed"].notna()
    if has_sectors.any():
        from scipy.stats import spearmanr

        rho = spearmanr(
            days[has_sectors.to_numpy()],
            joined.loc[has_sectors, "n_sectors_observed"].to_numpy(dtype=float),
        ).statistic
        log.info(
            "[candidate-bias] cross-check rho(baseline_days, n_sectors_observed) = %+.3f over %d",
            float(rho),
            int(has_sectors.sum()),
        )

    scores = joined["prob_mean"].to_numpy(dtype=float)
    payload = {
        "all_missions": _report(joined, scores, "all"),
        "by_mission": {
            mission: _report(part, part["prob_mean"].to_numpy(dtype=float), mission)
            for mission, part in joined.groupby("mission_view")
        },
        "reference_labelled_set": {
            "label_baseline_days_all": 0.278,
            "label_baseline_days_tess": 0.387,
            "incumbent_baseline_days_all": 0.238,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    log.info("[candidate-bias] wrote %s", args.out)


if __name__ == "__main__":
    main()
