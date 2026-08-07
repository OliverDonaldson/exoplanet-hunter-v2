"""Score a run onto a shard set, compare two prediction sets, or summarise one.

One entry point for all three, because they are the same question asked at
different stages and keeping them apart is how three different readings of the
incumbent came to exist. `score` produces a `predictions.parquet`; `compare`
reads two of them; `summarise` turns one into the `cv_summary.json` the
promotion gate reads.

**`summarise` is what lets the gate engage at all.** `evaluate_promotion` gates
on `per_mission[TESS]`, the live incumbent's summary has no such block, and the
re-baseline existed only as predictions — so every stage-2 decision was refused
(and before 2026-08-07, silently taken on pooled means). Slices are out-of-fold
only: the re-baselined set carries 243 zero-shot Kepler rows that are *all*
negatives, and pooling them into the out-of-fold Kepler figure measures a
population no model was asked about.

Coverage is reported before any metric. The stage 2(a) comparison matched 4,605
rows and looked complete while dropping all 527 K2 examples, because the
incumbent's run predates K2 entirely — so a mission the join loses is named
first, and `--strict` refuses the comparison outright.

The stratified cuts exist because the aggregate hid the mechanism. Quartiles are
cut per mission, so Kepler's top quartile is ~1,035 transits caught where TESS's
is ~89: "TESS looks flat" partly means "TESS never gets there". The absolute
bands put both missions on one cut, and the span-by-count cross separates a
resolution deficit from a data deficit. All three are the same measurement over
different binnings, so they share one primitive in `eval.comparison`.

    python pipeline/scripts/evaluate.py compare \
        --left  models/cv/<incumbent>/predictions.parquet \
        --right models/cv/<candidate>/predictions.parquet

    python pipeline/scripts/evaluate.py score \
        --run models/cv/<incumbent> --protocol zeroshot --mission K2 \
        --out results/incumbent_k2.parquet

    python pipeline/scripts/evaluate.py summarise \
        --predictions results/incumbent_rebaselined.parquet \
        --out models/cv/incumbent-rebaselined/cv_summary.json --exclude-unresolved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.eval.comparison import (
    band_labels,
    compare_prediction_sets,
    gap_table,
    paired_frame,
)
from exoplanet_hunter.utils.logging import get_logger
from exoplanet_hunter.validation.promotion import GATE_MISSION

log = get_logger(__name__)

#: The incumbent's prediction set predates these names.
LEGACY_COLUMNS = {"y_true": "label", "prob_calibrated": "score"}
QUARTILE_LABELS = ("Q1 fewest", "Q2", "Q3", "Q4 most")
TRANSIT_BANDS = ((0, 10), (10, 30), (30, 100), (100, 300), (300, np.inf))
#: Transit span is measured in bins of the *coarse* grid, so the number stays
#: comparable across runs whatever resolution each was built at.
REFERENCE_GLOBAL_BINS = 301
SPAN_BANDS = ((0, 4), (4, 8), (8, np.inf))
COUNT_BANDS = ((0, 100), (100, np.inf))
COVARIATES = ["tic_id", "mission", "observed_transit_count", "period", "duration"]


def load_predictions(path: Path, missions: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path).rename(columns=LEGACY_COLUMNS)
    return frame if "mission" in frame.columns else frame.merge(missions, on="tic_id", how="left")


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string(index=False, na_rep="—", float_format=lambda v: f"{v:.4f}"))


def cmd_compare(args: argparse.Namespace) -> None:
    scalars = pd.read_parquet(args.scalars)
    missions = scalars[["tic_id", "mission"]].drop_duplicates()
    covariates = scalars[COVARIATES].drop_duplicates("tic_id")

    left = load_predictions(args.left, missions)
    right = load_predictions(args.right, missions)
    result = compare_prediction_sets(left, right, strict=args.strict)

    print(f"\nleft   {args.left}\nright  {args.right}\n")
    print(result.coverage.report())
    if result.coverage.dropped:
        print(
            "\n*** the rows above are excluded from every number below. "
            "A mission at 0% shared is unmeasured, not equal. ***"
        )

    order = [m for m in result.left if m != "all"] + ["all"]
    show(
        "headline — TESS gates, the pool never does",
        pd.DataFrame(
            [
                {
                    "slice": mission,
                    "n": result.left[mission].n,
                    "n_pos": result.left[mission].n_positive,
                    "left": result.left[mission].roc_auc,
                    "right": result.right[mission].roc_auc,
                    "gap": result.left[mission].roc_auc - result.right[mission].roc_auc,
                    "left R@1%": result.left[mission].recall_at_1pct_fpr,
                    "right R@1%": result.right[mission].recall_at_1pct_fpr,
                }
                for mission in order
            ]
        ),
    )

    paired = paired_frame(left, right, covariates)
    paired["span"] = paired["duration"] / paired["period"] * REFERENCE_GLOBAL_BINS

    for mission in args.missions:
        rows = paired[paired["mission"] == mission].dropna(subset=["observed_transit_count"])
        if len(rows) < 40:
            log.info("[compare] %s: too few shared rows for a quartile split", mission)
            continue
        rows = rows.assign(
            quartile=pd.qcut(
                rows["observed_transit_count"].rank(method="first"),
                4,
                labels=list(QUARTILE_LABELS),
            )
        )
        show(
            f"{mission}: AUC gap by transits caught (quartiles cut within this mission)",
            gap_table(rows, "quartile", medians=["observed_transit_count"]),
        )

    banded = paired.dropna(subset=["observed_transit_count"]).assign(
        band=lambda f: band_labels(f["observed_transit_count"], TRANSIT_BANDS)
    )
    show(
        "transits caught — both missions on the same absolute cut",
        gap_table(banded, ["band", "mission"], min_rows=args.min_band_rows),
    )

    crossed = paired[np.isfinite(paired["span"]) & (paired["span"] > 0)].assign(
        span_band=lambda f: band_labels(f["span"], SPAN_BANDS),
        count_band=lambda f: band_labels(f["observed_transit_count"], COUNT_BANDS),
    )
    for mission in args.missions:
        cells = gap_table(
            crossed[crossed["mission"] == mission],
            ["span_band", "count_band"],
            min_rows=args.min_band_rows,
            medians=["span"],
        )
        if np.isfinite(cells["gap"]).any():
            show(
                f"{mission}: transit span (in {REFERENCE_GLOBAL_BINS}-bin units) x transits", cells
            )


def cmd_score(args: argparse.Namespace) -> None:
    from exoplanet_hunter.eval.scoring import Protocol, score_run

    protocol = Protocol(args.protocol)
    index = pd.read_parquet(args.shard_dir / "index.parquet")[["tic_id"]]
    scalars = pd.read_parquet(args.scalars)[["tic_id", "mission"]].drop_duplicates()
    missions = index.merge(scalars, on="tic_id", how="left")["mission"]

    subset = (missions == args.mission).to_numpy() if args.mission else None
    if subset is not None and not subset.any():
        raise SystemExit(f"no {args.mission} rows in {args.shard_dir}")

    fold_of = None
    if protocol is Protocol.OUT_OF_FOLD:
        reference = pd.read_parquet(args.run / "predictions.parquet")
        fold_of = reference.set_index("tic_id")["fold"]

    scored = score_run(
        args.run,
        args.shard_dir,
        labels=pd.read_parquet(args.labels),
        protocol=protocol,
        fold_of=fold_of,
        subset=subset,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.predictions.to_parquet(args.out, index=False)
    log.info(
        "[score] %s: %d rows under %s -> %s",
        args.run.name,
        len(scored.predictions),
        scored.protocol,
        args.out,
    )


def cmd_summarise(args: argparse.Namespace) -> None:
    from exoplanet_hunter.eval.scoring import summarise_scored

    scalars = pd.read_parquet(args.scalars)[["tic_id", "mission"]].drop_duplicates("tic_id")
    scored = load_predictions(args.predictions, scalars)
    if "protocol" not in scored.columns:
        if args.protocol is None:
            raise SystemExit(
                f"{args.predictions} carries no protocol column. Pass --protocol to declare "
                "how these rows were scored; guessing is what made a zero-shot number "
                "readable as a held-out one."
            )
        scored = scored.assign(protocol=args.protocol)

    summary = summarise_scored(
        scored, source=str(args.predictions), exclude_unresolved=args.exclude_unresolved
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    gate = summary["per_mission"][GATE_MISSION]
    log.info(
        "[summarise] %d of %d rows gate on %s: AUC %.4f, recall@1%%FPR %.3f -> %s",
        summary["provenance"]["n_gated"],
        summary["provenance"]["n_rows"],
        GATE_MISSION,
        gate["roc_auc"],
        gate["recall_at_1pct_fpr"],
        args.out,
    )
    for mission, metrics in summary["zero_shot"].items():
        log.info(
            "[summarise] %s: %d zero-shot rows reported, never gates (AUC %.4f)",
            mission,
            metrics["n"],
            metrics["roc_auc"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scalars", type=Path, default=Path("data/processed/viewset_scalars.parquet")
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="two prediction sets, per mission and stratified")
    compare.add_argument("--left", type=Path, required=True, help="usually the incumbent")
    compare.add_argument("--right", type=Path, required=True, help="the candidate")
    compare.add_argument("--missions", nargs="*", default=["Kepler", "TESS"])
    compare.add_argument("--min-band-rows", type=int, default=60)
    compare.add_argument(
        "--strict", action="store_true", help="refuse if the join drops a mission entirely"
    )
    compare.set_defaults(func=cmd_compare)

    score = sub.add_parser("score", help="a run's folds over a shard set")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--shard-dir", type=Path, default=Path("data/processed/tfrecords"))
    score.add_argument("--labels", type=Path, default=Path("data/labels/labels.parquet"))
    score.add_argument("--protocol", choices=["oof", "zeroshot"], required=True)
    score.add_argument("--mission", default=None, help="restrict to one mission")
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(func=cmd_score)

    summarise = sub.add_parser(
        "summarise", help="a gate-readable cv_summary.json from a scored prediction set"
    )
    summarise.add_argument("--predictions", type=Path, required=True)
    summarise.add_argument("--out", type=Path, required=True)
    summarise.add_argument(
        "--protocol",
        choices=["oof", "zeroshot"],
        default=None,
        help="declare the protocol when the frame carries no protocol column",
    )
    summarise.add_argument(
        "--exclude-unresolved",
        action="store_true",
        help=(
            "drop rows whose mission does not resolve — they were scored against a shard "
            "set the mission source does not cover, so no candidate can score them. Each "
            "excluded tic_id is logged and recorded in the summary's provenance block."
        ),
    )
    summarise.set_defaults(func=cmd_summarise)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
