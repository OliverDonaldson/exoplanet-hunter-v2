"""Re-score the served model on the current population, so a delta means the model.

The gate compares a candidate scored on today's catalogue against the
champion's summary from whenever it was trained. That difference is a model
effect and a population effect added together. This produces the missing half:
the same champion, measured now, on the rows the candidate is judged on.

Run from the repository root:

    # the served model, re-scored on the current shard set
    python pipeline/scripts/control_lane.py \
        --out models/cv/control-lane/cv_summary.json

    # check it computes what the original path computes (4.1d Check A). The
    # reference is regenerated TODAY, from the same shards and the same labels,
    # so the only difference left between the two sides is the code path:
    #   evaluate.py score --run <champion> --protocol oof --out <ref>.parquet
    #   evaluate.py summarise --predictions <ref>.parquet --protocol oof \
    #       --out <ref>/cv_summary.json --exclude-unresolved
    python pipeline/scripts/control_lane.py \
        --out models/cv/control-lane/cv_summary.json \
        --reproduces <ref>/cv_summary.json \
        --reproduces-predictions <ref>.parquet

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
    rows_reproduce,
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
            "a cv_summary.json produced by the original scoring path FROM THE SAME "
            "INPUTS ON THE SAME DAY. Every metric of every per-mission slice must agree "
            "to 1e-6: both sides then differ only in code path, which is the only thing "
            "this checks. Pointing it at a summary from another date makes it a test of "
            "whether the catalogue moved instead — the mistake 4.1c made and 4.1d fixed"
        ),
    )
    parser.add_argument(
        "--reproduces-predictions",
        type=Path,
        default=None,
        help=(
            "the predictions.parquet beside --reproduces. Checks membership, fold, label "
            "and score row by row rather than trusting slice means, which can agree while "
            "the individual objects the shortlist is made of do not"
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
        original = json.loads(args.reproduces.read_text())
        if drifted := reproduces(summary, original):
            raise SystemExit(
                f"the lane disagrees with {args.reproduces} on {len(drifted)} metric(s): "
                f"{'; '.join(drifted)}. Both measure the same weights over the same shards "
                "and the same labels, so the only thing left between them is the code path "
                "— the lane computes something other than what it claims to, and every "
                "delta built on it is void"
            )
        log.info("[control] matches %s on every slice", args.reproduces)
    if args.reproduces_predictions is not None:
        rows = pd.read_parquet(args.reproduces_predictions)
        if drifted := rows_reproduce(frame, rows):
            raise SystemExit(
                f"the lane disagrees with {args.reproduces_predictions} row by row: "
                f"{'; '.join(drifted)}. Slice means can agree while individual objects do "
                "not, and the shortlist this system exists to produce is made of individual "
                "objects, so this is the disagreement that matters"
            )
        log.info("[control] matches %s row for row", args.reproduces_predictions)

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
