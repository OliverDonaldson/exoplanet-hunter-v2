"""Offline control arm for a CV run directory — stage 7i.

Scores real hosts with **no injected transit** through a run directory's own
fold members and calibrator, and reports how many still pass threshold. A pass
means the model scored the *star*, not the transit.

    clean -> flatten -> inject_box_transit(depth=0) -> build_view_set
          -> write_viewset_shards -> make_viewset_dataset
          -> fold members + calibrator from the run directory

The shard round-trip is the construction rather than a shortcut: it puts the
control arm through the same parse and scalar-normalisation path training used,
instead of a second implementation that agrees with it by inspection.

Depth 0 makes `inject_box_transit` a no-op by construction, which is the point —
the model is shown a real light curve at an ephemeris where nothing happens. It
is called anyway so this path and the graded injection path differ in the depth
argument alone.

Usage (from the repository root):

    python pipeline/scripts/control_arm.py \
        --run-dir models/cv/branches-20260808-rebaseline --hosts 80

Both limits pre-registered on 2026-08-09 apply to every number this writes and
are recorded into the output JSON rather than left to a reader's memory:

1. `detection` / `ghost` run **masked** — no DV report exists at a synthetic
   ephemeris, so `dv_usable` is False for every row. That is a real difference
   from how 56% of training rows were built.
2. This does **not** restore comparability with the live path's 26.4%. The
   incumbent must be re-measured through this same harness and the comparison
   made within protocol.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.viewset_io import stack_view_sets
from exoplanet_hunter.datasets.viewset_tfrecords import (
    FEATURE_COLUMNS,
    MASK_COLUMNS,
    write_viewset_shards,
)
from exoplanet_hunter.eval.control_arm import (
    baseline_matched_hosts,
    control_arm_rate,
    fold_assignment,
    operating_points,
)
from exoplanet_hunter.eval.injection_recovery import inject_box_transit
from exoplanet_hunter.preprocess import clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.preprocess.viewset import build_view_set
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

#: Periods to place the synthetic ephemeris at. Three rather than one so a
#: pass cannot be an artefact of a single folding.
DEFAULT_PERIODS = (3.0, 7.0, 12.0)
#: Central transit of a Sun-like star, scaled from Earth's 13 h.
SOLAR_DURATION_HOURS = 13.0


def transit_duration_hours(period_days: float) -> float:
    return SOLAR_DURATION_HOURS * (period_days / 365.25) ** (1.0 / 3.0)


def control_scalars(rows: list[dict]) -> pd.DataFrame:
    """The scalar frame for a control-arm shard set.

    Every DV-derived column is NaN and `dv_usable` is False — limit 1 above.
    NaN rather than 0.0 because the reader imputes NaN to the fold's own fitted
    centre, where it carries no information; a zero would land at a real
    percentile of the column and be read as a weak measurement.

    Written through `FEATURE_COLUMNS` / `MASK_COLUMNS` rather than a literal
    list, so a branch added to the schema cannot leave this path writing a
    shorter vector that still passes every downstream gate.
    """
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in MASK_COLUMNS:
        if column not in frame.columns:
            frame[column] = False
    return frame


def build_host_views(
    tic_id: int, fits_path: Path, period: float, sigma_clip: float, window: int, polyorder: int
) -> tuple[object, dict]:
    """Views and scalars for one host at a synthetic ephemeris, depth 0."""
    import lightkurve as lk

    raw = lk.read(str(fits_path))
    cleaned = clean_lightcurve(raw, sigma_clip=sigma_clip)
    flat = flatten_lightcurve(cleaned, window_length=window, polyorder=polyorder)

    time = np.asarray(flat.time.value, dtype=float)
    flux = np.asarray(flat.flux.value, dtype=float)
    duration_d = transit_duration_hours(period) / 24.0
    t0 = float(np.nanmin(time)) + 0.5 * period

    # Depth 0: a no-op by construction, called so this path differs from the
    # graded injection path in exactly one argument.
    flat.flux[:] = inject_box_transit(time, flux, period, t0, duration_d, depth=0.0)

    views = build_view_set(flat, period=period, t0=t0, duration=duration_d, raw_lc=raw)
    observed = int(views.observed_transit_count)
    expected = int(views.expected_transit_count)
    scalars = {
        "tic_id": tic_id,
        "mission": "TESS",
        "observed_transit_count": observed,
        "expected_transit_count": expected,
        "transit_completeness": observed / expected if expected else 0.0,
        "secondary_phase": float(views.secondary_phase),
        "period": period,
        "dv_usable": False,
        "has_ruwe": False,
    }
    return views, scalars


def score_through_run(
    run_dir: Path, shard_dir: Path, index: pd.DataFrame, folds: dict
) -> pd.Series:
    """Calibrated score per row, each from the fold that held its host out."""
    import joblib
    import tensorflow as tf

    from exoplanet_hunter.datasets.viewset_pipeline import (
        make_viewset_dataset,
        parse_viewset_shards,
    )
    from exoplanet_hunter.datasets.viewset_tfrecords import list_shards, load_metadata

    # Imported for its side effect: PresenceFlag, PickColumns, StackViews and
    # MaskedTransitPool are registered with @register_keras_serializable at
    # import time, and load_model resolves them from that registry. Without this
    # the checkpoint deserialises to a bare Functional with an empty config and
    # raises "Could not locate class 'StackViews'".
    from exoplanet_hunter.models import cnn_branches  # noqa: F401

    metadata = load_metadata(shard_dir)
    shards = list_shards(shard_dir)
    base = parse_viewset_shards(shards, metadata)
    scores = pd.Series(np.nan, index=index.index, dtype=float)

    for fold in sorted(set(folds.values())):
        fold_dir = run_dir / f"fold_{fold}"
        bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
        members = sorted(fold_dir.glob("model_*_cnn_branches.keras"))
        if not members:
            raise FileNotFoundError(f"{fold_dir} carries no member checkpoints")

        stream = make_viewset_dataset(
            shards,
            metadata,
            base=base,
            scalar_constants=bundle["scalar_constants"],
            batch_size=32,
            shuffle=False,
            with_tic_id=True,
        )
        raw = np.mean(
            [
                np.concatenate(
                    [
                        tf.keras.models.load_model(str(m), compile=False)
                        .predict(inputs, verbose=0)
                        .ravel()
                        for inputs, _, _ in stream
                    ]
                )
                for m in members
            ],
            axis=0,
        )
        tics = np.concatenate([t.numpy() for _, _, t in stream])
        calibrated = bundle["calibrator"].predict(raw)

        # Only the rows this fold owns. Every row is scored by exactly one fold,
        # so a row left NaN below means its host had no fold and was dropped.
        wanted = {tic for tic, f in folds.items() if f == fold}
        written = 0
        for tic, value in zip(tics, calibrated, strict=True):
            mask = (index["tic_id"] == int(tic)).to_numpy()
            if int(tic) in wanted and mask.any():
                scores.loc[index.index[mask]] = float(value)
                written += int(mask.sum())
        # The count written here, not len(wanted): that is the fold's whole
        # membership across the run and would report ~1,085 for a control arm
        # of eight hosts.
        log.info("[control-arm] fold %d scored %d control-arm row(s)", fold, written)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/labels.parquet"))
    parser.add_argument(
        "--viewset-index", type=Path, default=Path("data/processed/viewset_tfrecords/index.parquet")
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/control_arm"))
    parser.add_argument("--hosts", type=int, default=80, help="target matched host count")
    parser.add_argument("--n-strata", type=int, default=4)
    parser.add_argument("--periods", type=float, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--max-host-depth-ppm", type=float, default=3000.0)
    parser.add_argument("--sigma-clip", type=float, default=5.0)
    parser.add_argument("--window-length", type=int, default=401)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.run_dir / "predictions.parquet")
    points = operating_points(predictions)
    folds = fold_assignment(predictions)
    log.info(
        "[control-arm] %s: shortlist cut %.4f, F1 cut %.4f (from %d TESS rows, %d positive)",
        args.run_dir.name,
        points.shortlist,
        points.f1_optimal,
        points.n,
        points.n_positive,
    )

    labels = pd.read_parquet(args.labels)
    labels = labels[labels["mission"] == "TESS"]
    labels = labels[labels["depth"].fillna(np.inf) * 1e6 <= args.max_host_depth_ppm]
    cached = {int(p.stem.split("_")[1]) for p in args.raw.glob("tic_*.fits")}
    labels = labels[labels["tic_id"].isin(cached)]
    # Only hosts this run can route out-of-fold; the rest are dropped, per the
    # pre-registered protocol, rather than scored by an averaged ensemble.
    routable = labels[labels["tic_id"].isin(folds)]
    log.info(
        "[control-arm] %d cached TESS hosts, %d routable out-of-fold (%d dropped)",
        len(labels),
        len(routable),
        len(labels) - len(routable),
    )

    scalars_index = pd.read_parquet(args.viewset_index)
    joined = routable.merge(
        scalars_index[["tic_id", "expected_transit_count"]].drop_duplicates("tic_id"),
        on="tic_id",
        how="inner",
        validate="one_to_one",
    )
    matched = baseline_matched_hosts(
        joined,
        per_label_per_stratum=max(1, args.hosts // (2 * args.n_strata)),
        n_strata=args.n_strata,
        seed=args.seed,
    )
    log.info("[control-arm] %s", matched.report())
    if matched.n == 0:
        raise SystemExit("no matched hosts — relax --max-host-depth-ppm or --n-strata")

    view_sets, rows = [], []
    # `to_dict("records")` rather than `itertuples`: attribute access on a
    # namedtuple row types as the union of every column dtype, so int()/float()
    # over it cannot type-check. Records give plain dicts.
    for host in matched.hosts.to_dict("records"):
        tic_id = int(host["tic_id"])
        fits = next(args.raw.glob(f"tic_{tic_id}_*.fits"), None) or next(
            args.raw.glob(f"tic_{tic_id}.fits"), None
        )
        if fits is None:
            log.warning("[control-arm] TIC %d has no cached FITS — skipped", tic_id)
            continue
        for period in args.periods:
            try:
                views, scalars = build_host_views(
                    tic_id, fits, period, args.sigma_clip, args.window_length, args.polyorder
                )
            except Exception as exc:
                log.warning("[control-arm] TIC %d P=%.1f failed: %s", tic_id, period, exc)
                continue
            scalars["label"] = int(host["label"])
            scalars["baseline_days"] = float(host["baseline_days"])
            scalars["stratum"] = int(host["stratum"])
            view_sets.append(views)
            rows.append(scalars)
            log.info("[control-arm] built TIC %d P=%.1f d", tic_id, period)

    if not view_sets:
        raise SystemExit("no host views built")
    arrays = stack_view_sets(view_sets, control_scalars(rows))
    problems = arrays.validate()
    if problems:
        raise ValueError(f"control-arm view set is malformed: {problems}")

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir = Path(tmp) / "shards"
        write_viewset_shards(arrays, shard_dir)
        scored = score_through_run(args.run_dir, shard_dir, arrays.scalars, folds)

    frame = arrays.scalars.copy()
    frame["score"] = scored.to_numpy()
    unscored = int(frame["score"].isna().sum())
    if unscored:
        log.warning("[control-arm] %d row(s) unscored and dropped", unscored)
        frame = frame[frame["score"].notna()]

    results = {
        "run_dir": str(args.run_dir),
        "n_rows": len(frame),
        "n_hosts": int(frame["tic_id"].nunique()),
        "n_unscored_dropped": unscored,
        "matched": {
            "n_strata_used": matched.n_strata_used,
            "n_strata_dropped": matched.n_strata_dropped,
            "n_available": matched.n_available,
        },
        "operating_points": points.as_dict(),
        "limits": {
            "dv_masked": "detection/ghost run masked; no DV report at a synthetic ephemeris",
            "comparability": (
                "not comparable with the live path's 26.4%; the incumbent must be "
                "re-measured through this harness and compared within protocol"
            ),
        },
    }
    for name, cut in (("shortlist", points.shortlist), ("f1_optimal", points.f1_optimal)):
        rate = control_arm_rate(
            frame["score"].to_numpy(),
            frame["label"].to_numpy(),
            threshold=cut,
            threshold_name=name,
        )
        results[name] = rate.as_dict()
        log.info(
            "[control-arm] %-10s cut %.4f -> pass %.3f  (planet %.3f, FP %.3f, split %+.3f)",
            name,
            cut,
            rate.overall,
            rate.planet_hosts,
            rate.fp_hosts,
            rate.split,
        )

    stem = args.run_dir.name
    frame.to_parquet(args.out / f"{stem}.parquet", index=False)
    (args.out / f"{stem}.json").write_text(json.dumps(results, indent=2))
    log.info("[control-arm] wrote %s", args.out / f"{stem}.json")


if __name__ == "__main__":
    main()
