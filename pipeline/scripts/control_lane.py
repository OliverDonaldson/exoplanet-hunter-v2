"""Re-score the served model on the current population, so a delta means the model.

The gate compares a candidate scored on today's catalogue against the
champion's summary from whenever it was trained. That difference is a model
effect and a population effect added together. This produces the missing half:
the same champion, measured now, on the rows the candidate is judged on.

Run from the repository root:

    # the served model, re-scored on the current shard set
    python pipeline/scripts/control_lane.py \
        --out models/cv/control-lane/cv_summary.json

    # check it reproduces a stored measurement of the same model
    python pipeline/scripts/control_lane.py \
        --out models/cv/control-lane/cv_summary.json \
        --reproduces models/cv/incumbent-rebaselined/cv_summary.json

The registry is never modified. This scores and summarises; deciding anything is
`promotion_gate.py`'s job, and it reads the summary written here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from exoplanet_hunter.eval.control_lane import (
    assert_gateable,
    population_overlap,
    reproduces,
)
from exoplanet_hunter.utils import get_logger
from exoplanet_hunter.validation import load_registry

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="the run to re-score; defaults to whatever the registry currently serves",
    )
    parser.add_argument("--shard-dir", type=Path, default=Path("data/processed/tfrecords"))
    parser.add_argument("--labels", type=Path, default=Path("data/tables/labels/labels.parquet"))
    parser.add_argument(
        "--scalars", type=Path, default=Path("data/processed/viewset_scalars.parquet")
    )
    parser.add_argument("--out", type=Path, required=True, help="where to write cv_summary.json")
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="also keep the scored rows; defaults to predictions.parquet beside --out",
    )
    parser.add_argument(
        "--reproduces",
        type=Path,
        default=None,
        help=(
            "a stored summary of this same model to check against. Same weights, same "
            "folds, the same rows — a disagreement beyond floating point means the "
            "re-score is not measuring the model the registry names, and the run fails"
        ),
    )
    args = parser.parse_args()

    from exoplanet_hunter.eval.scoring import Protocol, score_run, summarise_scored

    run = args.run
    if run is None:
        registry = load_registry(args.models_dir)
        if registry is None:
            raise SystemExit(
                f"no registry under {args.models_dir}, so there is no served model to "
                "re-score. Pass --run to name one explicitly"
            )
        run = args.models_dir.parent / registry["cv_dir"]
        log.info("[control] re-scoring the served run %s", registry["run_id"][:8])

    # The fold that held each row out. Scoring a row with a fold that trained on
    # it is not a held-out measurement, and the resulting number would flatter
    # the champion in a comparison built to be fair to the candidate.
    fold_of = pd.read_parquet(run / "predictions.parquet").set_index("tic_id")["fold"]

    index = pd.read_parquet(args.shard_dir / "index.parquet")
    overlap = population_overlap(set(fold_of.index), set(index["tic_id"]))
    log.info("[control] %s", overlap)

    # `score_run` under OUT_OF_FOLD keeps only rows it has a fold for, which is
    # exactly the shared population — the rows added since this model trained
    # have no fold that held them out and are excluded by construction.
    scored = score_run(
        run,
        args.shard_dir,
        labels=pd.read_parquet(args.labels),
        protocol=Protocol.OUT_OF_FOLD,
        fold_of=fold_of,
    )
    frame = scored.predictions.assign(protocol=scored.protocol)
    missions = pd.read_parquet(args.scalars)[["tic_id", "mission"]].drop_duplicates("tic_id")
    frame = frame.merge(missions, on="tic_id", how="left")

    summary = summarise_scored(frame, source=str(run), exclude_unresolved=True)
    summary["control_lane"] = {
        "run": run.name,
        "shard_dir": str(args.shard_dir),
        "shared": overlap.shared,
        "added_since": overlap.added,
        "dropped_since": overlap.dropped,
        "covered": overlap.covered,
        "note": (
            "the served model re-scored out-of-fold on the current shard set. Rows added "
            "since it trained are excluded: it never saw them, so scoring them would "
            "average every fold and hand this model an ensemble the candidate does not get"
        ),
    }

    # Both guards run before anything is written. A summary on disk is a summary
    # something will read, and a caller that finds one has no way to know it was
    # produced by a run that failed its own checks.
    assert_gateable(summary)
    if args.reproduces is not None:
        stored = json.loads(args.reproduces.read_text())
        if drifted := reproduces(summary, stored):
            raise SystemExit(
                "the re-scored champion does not reproduce its stored measurement on "
                f"{len(drifted)} metric(s): {'; '.join(drifted)}. Same weights and the same "
                "rows should give the same numbers, so the lane is measuring something "
                "other than the model the registry names and every delta from it is void"
            )
        log.info("[control] reproduces %s exactly", args.reproduces)

    predictions_out = args.predictions_out or args.out.parent / "predictions.parquet"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(predictions_out, index=False)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    gate = summary["per_mission"]["TESS"]
    log.info(
        "[control] TESS %d rows: AUC %.4f, recall@1%%FPR %.4f -> %s",
        gate["n"],
        gate["roc_auc"],
        gate["recall_at_1pct_fpr"],
        args.out,
    )


if __name__ == "__main__":
    main()
