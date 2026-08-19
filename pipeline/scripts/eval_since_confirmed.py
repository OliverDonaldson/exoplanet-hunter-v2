"""Prospective evaluation: score held-out candidates whose labels arrived later.

The training build wrote its unlabelled PCs to data/tables/labels/candidates.parquet;
the refreshed TOI catalogue now carries final dispositions for some of them.
Those targets were never trained on and their labels post-date the model —
the closest thing to a real-world test the archive offers.

Runs the full live scoring path per target (download + preprocess + ensemble),
so expect ~1-2 min each on first contact. Results land in
results/since_confirmed.parquet; rerunning skips already-scored targets.

Usage (from the repository root):

    python pipeline/scripts/eval_since_confirmed.py [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.scoring import NoLightCurveError, TargetScorer
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

POSITIVE = {"CP", "KP"}
NEGATIVE = {"FP", "FA"}


def flipped_holdout(holdout_path: Path, catalogue_path: Path) -> pd.DataFrame:
    """Training-time PCs that now carry a final disposition (same TIC+period).

    The refreshed disposition is renamed **before** the merge rather than left to
    pandas' `suffixes`. Both frames carry a `disposition` column, so an unrenamed
    merge leaves `disposition` holding the *training-time* value — which is `PC`
    for every held-out candidate by definition — and a `y_true` read from it is
    zero on every row. That is not a hypothetical: it is what produced the two
    stored rows of 2026-07-20, one of them a confirmed planet labelled 0, and the
    same column-collision that once dropped the transit counts past every gate.
    """
    pc = pd.read_parquet(holdout_path)
    cat = pd.read_parquet(catalogue_path, columns=["tic_id", "period_days", "disposition"]).dropna()
    cat = cat[cat["disposition"].isin(POSITIVE | NEGATIVE)].rename(
        columns={"disposition": "disposition_now"}
    )

    merged = pc.merge(cat, on="tic_id")
    same_planet = np.abs(merged["period_days"] - merged["period"]) / merged["period"] < 0.01
    merged = merged[same_planet & merged["period"].gt(0) & merged["duration"].gt(0)]
    merged["y_true"] = merged["disposition_now"].isin(POSITIVE).astype(int)
    merged = merged.drop_duplicates("tic_id")

    # A single-class set cannot be evaluated: AUC is undefined and recall is
    # either 0 or 1 by construction, both of which read as a measurement.
    if len(merged) and merged["y_true"].nunique() < 2:
        raise ValueError(
            f"all {len(merged)} prospective targets carry the same label "
            f"({merged['disposition_now'].unique().tolist()}) — there is nothing to rank, "
            "and a ranking metric over one class is a number rather than a measurement"
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout", type=Path, default=Path("data/tables/labels/candidates.parquet")
    )
    parser.add_argument(
        "--catalogue", type=Path, default=Path("data/tables/catalogue/candidates.parquet")
    )
    parser.add_argument("--out", type=Path, default=Path("results/since_confirmed.parquet"))
    parser.add_argument("--limit", type=int, default=None, help="score at most N targets")
    args = parser.parse_args()

    targets = flipped_holdout(args.holdout, args.catalogue)
    log.info(
        "[since-confirmed] %d held-out candidates have final dispositions (%d planets, %d FPs)",
        len(targets),
        int(targets["y_true"].sum()),
        int((1 - targets["y_true"]).sum()),
    )

    done = pd.read_parquet(args.out) if args.out.exists() else pd.DataFrame(columns=["tic_id"])
    todo = targets[~targets["tic_id"].isin(done["tic_id"])]
    if args.limit:
        todo = todo.head(args.limit)

    scorer = TargetScorer(
        models_dir=Path("models"),
        data_raw=Path("data/raw/tess/lightcurves"),
        candidates_path=args.catalogue,
    )

    rows = [] if done.empty else [done]
    scored = 0
    for record in todo.itertuples():
        try:
            outcome = scorer.score(
                int(record.tic_id),
                period_days=float(record.period),
                t0_btjd=float(record.t0),
                duration_hours=float(record.duration) * 24.0,
                n_mc=20,
            )
        except NoLightCurveError as exc:
            log.warning("[since-confirmed] TIC %d skipped: %s", record.tic_id, exc)
            continue
        except Exception as exc:
            log.warning("[since-confirmed] TIC %d failed: %s", record.tic_id, exc)
            continue
        scored += 1
        rows.append(
            pd.DataFrame(
                [
                    {
                        "tic_id": int(record.tic_id),
                        "name": record.name,
                        "disposition_now": record.disposition_now,
                        "y_true": int(record.y_true),
                        "prob_calibrated": outcome.prob_calibrated,
                        "prob_std": outcome.prob_std,
                        "threshold": outcome.threshold,
                    }
                ]
            )
        )
        result = pd.concat(rows, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.out, index=False)  # checkpoint after every target
        log.info(
            "[since-confirmed] %d/%d  TIC %d (%s) -> %.3f",
            scored,
            len(todo),
            record.tic_id,
            record.disposition_now,
            outcome.prob_calibrated,
        )

    result = pd.read_parquet(args.out)
    if result["y_true"].nunique() == 2:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        thr = float(result["threshold"].iloc[0])
        acc = float(((result["prob_calibrated"] >= thr) == result["y_true"]).mean())
        log.info(
            "[since-confirmed] n=%d  AUC=%.4f  Brier=%.4f  acc@%.2f=%.3f",
            len(result),
            roc_auc_score(result["y_true"], result["prob_calibrated"]),
            brier_score_loss(result["y_true"], result["prob_calibrated"]),
            thr,
            acc,
        )


if __name__ == "__main__":
    main()
