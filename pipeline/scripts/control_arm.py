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
   champion must be re-measured through this same harness and the comparison
   made within protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

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
from exoplanet_hunter.features.aux import LEGACY_AUX_DIM
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
    tic_id: int,
    fits_path: Path,
    period: float,
    sigma_clip: float,
    window: int,
    polyorder: int,
    momentum_dumps: np.ndarray | None = None,
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

    # `momentum_dumps` is passed and the DV inputs are not, and the asymmetry is
    # the pre-registered limit read correctly rather than an exception to it.
    # Limit 1 masks the DV columns because **no DV report exists at a synthetic
    # ephemeris**. A reaction-wheel desaturation is not a DV product: it happened
    # to the spacecraft at a real time, and folding it on a synthetic period is
    # exactly as meaningful as folding this star's flux on one. Withholding it
    # would gate the branch off and quietly measure a model the run did not build.
    views = build_view_set(
        flat,
        period=period,
        t0=t0,
        duration=duration_d,
        raw_lc=raw,
        momentum_dumps=momentum_dumps,
    )
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


def assert_stream_aligned(tics: np.ndarray, index: pd.DataFrame) -> None:
    """Raise unless the shard stream came back in the shard index's own order.

    Extracted from `score_through_run` so it can be made to fire without a
    trained model on disk. An unshuffled stream is *supposed* to match the index
    row for row; when it does not, every score is attached to the wrong target
    and the pass rate stays entirely believable, which is this project's
    defining failure mode.
    """
    seen = np.asarray(tics).astype(np.int64)
    expected = index["tic_id"].to_numpy(np.int64)
    if len(seen) != len(expected):
        raise RuntimeError(
            f"shard stream returned {len(seen)} row(s) for an index of {len(expected)}"
        )
    if not np.array_equal(seen, expected):
        wrong = int((seen != expected).sum())
        raise RuntimeError(
            f"shard stream is not aligned with the shard index: {wrong} of {len(seen)} "
            "row(s) differ, so every score would be attached to the wrong target"
        )


def score_through_run(
    run_dir: Path, shard_dir: Path, folds: dict
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score the shard set through a run directory's own fold members.

    Returns the shard **index** and a score per row, aligned positionally.

    Alignment is against the index the writer actually produced, not the frame
    handed to it: `write_viewset_shards` permutes rows before sharding, so the
    two are in different orders. Matching on `tic_id` instead — which an earlier
    version did — silently collapses a host's three periods onto one score,
    because every row of that host matches and the last write wins. The pass
    rate stays entirely plausible while two thirds of the measurement is
    overwritten.
    """
    import joblib
    import tensorflow as tf

    from exoplanet_hunter.datasets.viewset_pipeline import (
        make_viewset_dataset,
        parse_viewset_shards,
    )
    from exoplanet_hunter.datasets.viewset_tfrecords import list_shards, load_index, load_metadata

    # Imported for its side effect: PresenceFlag, PickColumns, StackViews and
    # MaskedTransitPool are registered with @register_keras_serializable at
    # import time, and load_model resolves them from that registry. Without this
    # the checkpoint deserialises to a bare Functional with an empty config and
    # raises "Could not locate class 'StackViews'".
    from exoplanet_hunter.models import cnn_branches  # noqa: F401

    metadata = load_metadata(shard_dir)
    shards = list_shards(shard_dir)
    index = load_index(shard_dir).reset_index(drop=True)
    base = parse_viewset_shards(shards, metadata)
    scores = np.full(len(index), np.nan, dtype=float)

    def stream() -> tf.data.Dataset:
        return make_viewset_dataset(
            shards,
            metadata,
            base=base,
            scalar_constants=bundle["scalar_constants"],
            batch_size=64,
            shuffle=False,
            with_tic_id=True,
        )

    for fold in sorted(set(folds.values())):
        fold_dir = run_dir / f"fold_{fold}"
        bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
        members = sorted(fold_dir.glob(BRANCH_CHECKPOINT))
        if not members:
            raise FileNotFoundError(f"{fold_dir} carries no member checkpoints")

        per_member = []
        tics: np.ndarray | None = None
        for path in members:
            model = tf.keras.models.load_model(str(path), compile=False)
            preds, seen = [], []
            for inputs, _, tic in stream():
                preds.append(np.asarray(model.predict_on_batch(inputs)).ravel())
                if tics is None:
                    seen.append(tic.numpy())
            per_member.append(np.concatenate(preds))
            if tics is None:
                # Captured on the first member's pass rather than a fourth
                # sweep of the stream purely to read identities back.
                tics = np.concatenate(seen)

        assert tics is not None
        assert_stream_aligned(tics, index)

        calibrated = bundle["calibrator"].predict(np.mean(per_member, axis=0))
        fold_of_row = index["tic_id"].map(lambda t: folds.get(int(t), -1)).to_numpy(int)
        owned = fold_of_row == fold
        scores[owned] = calibrated[owned]
        log.info("[control-arm] fold %d scored %d control-arm row(s)", fold, int(owned.sum()))
    return index, scores


#: What a run directory's fold_0 contains decides which lane scores it. A flag
#: would let a dual-view run be scored down the eleven-view path, which fails
#: with a shape error at best and a wrong number at worst.
BRANCH_CHECKPOINT = "model_*_cnn_branches.keras"
DUALVIEW_CHECKPOINT = "cnn_dualview.keras"
#: A multi-member run numbers its checkpoints and the bare name stops existing;
#: the branch lane has always globbed, the dual-view lane matched an exact name
#: and so became unscoreable the moment `n_models_per_fold > 1` landed on it.
DUALVIEW_CHECKPOINT_GLOB = "model_*_cnn_dualview.keras"


def dualview_members(fold_dir: Path) -> list[Path]:
    """Every dual-view checkpoint in one fold, numbered form preferred.

    Runs made before multi-member training existed still carry the bare
    `cnn_dualview.keras` and are still found and served by path, so both
    layouts are accepted rather than migrating the old ones.
    """
    numbered = sorted(fold_dir.glob(DUALVIEW_CHECKPOINT_GLOB))
    if numbered:
        return numbered
    legacy = fold_dir / DUALVIEW_CHECKPOINT
    return [legacy] if legacy.exists() else []


def run_kind(run_dir: Path) -> str:
    """`"branch"` or `"dualview"`, from the checkpoints actually on disk."""
    fold_0 = run_dir / "fold_0"
    has_branch = bool(list(fold_0.glob(BRANCH_CHECKPOINT)))
    has_dualview = bool(dualview_members(fold_0))
    if has_branch and has_dualview:
        raise ValueError(f"{fold_0} carries both checkpoint kinds; cannot choose a lane")
    if has_branch:
        return "branch"
    if has_dualview:
        return "dualview"
    raise FileNotFoundError(
        f"{fold_0} carries neither {BRANCH_CHECKPOINT} nor "
        f"{DUALVIEW_CHECKPOINT_GLOB} / {DUALVIEW_CHECKPOINT}"
    )


def dualview_aux_dim(run_dir: Path) -> int:
    """How many aux features this run's own fitted pipeline expects.

    Not a constant. The champion serves the 9-dim legacy layout, but any
    dual-view run trained since the vetting features landed expects 13, and
    handing its imputer a 9-dim row raises inside sklearn after the whole view
    build has already been paid for. The run's calibration bundle is the
    authority on its own width, so it is asked rather than assumed.
    """
    import joblib

    bundle = joblib.load(run_dir / "fold_0" / "cnn_calibrator.joblib")
    pipeline = bundle.get("aux_pipeline")
    if pipeline is None:
        return LEGACY_AUX_DIM
    width = getattr(pipeline, "n_features_in_", None)
    if width is None:
        raise ValueError(
            f"{run_dir}/fold_0 has an aux pipeline that does not declare "
            "n_features_in_; its feature width cannot be established"
        )
    return int(width)


def build_host_dualview(
    fits_path: Path,
    period: float,
    stellar: dict,
    sigma_clip: float,
    window: int,
    polyorder: int,
    aux_dim: int = LEGACY_AUX_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global view, local view and the aux row for a dual-view lane.

    The same host, the same cleaning and flattening and the same synthetic
    ephemeris as the branch lane — only the view builder and the feature layout
    differ, because that is exactly what distinguishes the two models.

    `depth` and `catalogue_snr` are left unset, so the fitted aux pipeline
    imputes them. That is not a shortcut: it is what the live path does for an
    ad-hoc target ("unknown for an ad-hoc target (as in training)",
    `service.py`), and a synthetic ephemeris has no catalogue row to read either
    from. Inventing a depth of 0 would assert "measured, and flat", which the
    imputer would then treat as a real value.
    """
    import lightkurve as lk

    from exoplanet_hunter.features.aux import build_aux_row
    from exoplanet_hunter.features.centroid import extract_centroid_offset
    from exoplanet_hunter.preprocess.views import build_views

    raw = lk.read(str(fits_path))
    cleaned = clean_lightcurve(raw, sigma_clip=sigma_clip)
    flat = flatten_lightcurve(cleaned, window_length=window, polyorder=polyorder)

    time = np.asarray(flat.time.value, dtype=float)
    flux = np.asarray(flat.flux.value, dtype=float)
    duration_d = transit_duration_hours(period) / 24.0
    t0 = float(np.nanmin(time)) + 0.5 * period
    flat.flux[:] = inject_box_transit(time, flux, period, t0, duration_d, depth=0.0)

    views = build_views(flat, period, t0, duration_d)
    # Computed rather than left NaN: the live path computes it, and omitting it
    # would hand the champion a thinner feature vector than it serves with.
    centroid_snr = float(extract_centroid_offset(raw, period, t0, duration_d))
    # Indices 9-12 of the 13-dim layout — the odd/even, secondary and duration
    # statistics — are left unset for the same reason as depth: a transit-free
    # synthetic light curve has no such measurement, and the fitted pipeline
    # imputes them exactly as it did in training.
    aux = build_aux_row(
        aux_dim,
        teff=stellar.get("teff"),
        radius=stellar.get("radius"),
        logg=stellar.get("logg"),
        tmag=stellar.get("tmag"),
        depth=None,
        duration=duration_d,
        period=period,
        catalogue_snr=None,
        centroid_snr=centroid_snr,
    )
    return (
        np.asarray(views.global_view, dtype=np.float32),
        np.asarray(views.local_view, dtype=np.float32),
        aux,
    )


def score_through_dualview(run_dir: Path, rows: list[dict], folds: dict) -> np.ndarray:
    """Calibrated score per row through the champion's own fold checkpoints.

    No shard round-trip here, and the asymmetry with the branch lane is
    deliberate rather than a corner cut. The branch model's scalar
    normalisation lives *in* the shard pipeline, so writing and reading a shard
    is the only way to reproduce it; the dual-view model's aux normalisation is
    a fitted sklearn pipeline carried in its own bundle, so replaying that
    pipeline is the equivalent fidelity guarantee. Both lanes reuse the exact
    transform their model was trained with.
    """
    import joblib
    import tensorflow as tf

    scores = np.full(len(rows), np.nan, dtype=float)
    for fold in sorted(set(folds.values())):
        fold_dir = run_dir / f"fold_{fold}"
        bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
        members = dualview_members(fold_dir)
        if not members:
            raise FileNotFoundError(f"{fold_dir} carries no dual-view checkpoint")
        pipeline = bundle.get("aux_pipeline")

        owned = [i for i, r in enumerate(rows) if folds.get(int(r["tic_id"])) == fold]
        if not owned:
            log.info("[control-arm] fold %d owns no control-arm row", fold)
            continue
        inputs = {
            "global_view": np.stack([rows[i]["global_view"] for i in owned])[..., None],
            "local_view": np.stack([rows[i]["local_view"] for i in owned])[..., None],
        }
        if pipeline is not None:
            inputs["aux_features"] = pipeline.transform(
                np.stack([rows[i]["aux"] for i in owned]).astype(np.float32)
            ).astype(np.float32)
        # Average the members' RAW scores, then calibrate once — the order the
        # trainer used (`val_score = val_members.mean(axis=0)` before the Platt
        # fit), so the calibrator is applied to the quantity it was fitted on.
        # Calibrating per member and averaging afterwards would push a fitted
        # sigmoid through an average it never saw.
        #
        # Deterministic pass, as the live path does for the headline: the
        # calibrators were fitted on deterministic scores, and feeding them MC
        # means costs ~0.08 ECE.
        per_member = [
            np.asarray(
                tf.keras.models.load_model(str(path), compile=False)(inputs, training=False)
            ).ravel()
            for path in members
        ]
        scores[owned] = bundle["calibrator"].predict(np.mean(per_member, axis=0))
        log.info(
            "[control-arm] fold %d scored %d control-arm row(s) through %d member(s)",
            fold,
            len(owned),
            len(members),
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/tables/labels/labels.parquet"))
    parser.add_argument(
        "--viewset-index", type=Path, default=Path("data/processed/viewset_tfrecords/index.parquet")
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/tess/lightcurves"))
    parser.add_argument(
        "--momentum-dumps",
        type=Path,
        default=None,
        help="flagged cadence table (default data/tables/momentum_dumps.parquet). Unlike the "
        "DV columns this is NOT masked by limit 1: a desaturation is a real event at a real "
        "time, not a DV product that cannot exist at a synthetic ephemeris",
    )
    parser.add_argument("--out", type=Path, default=Path("results/control_arm"))
    parser.add_argument(
        "--hosts",
        type=int,
        default=80,
        help=(
            "target total matched hosts; divided evenly over 2 labels x --n-strata. "
            "A stratum short of that many of either label contributes what it has, so "
            "the draw is usually smaller than this — the report says by how much"
        ),
    )
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=None,
        help="hosts per label per stratum, overriding --hosts. Set high to take the pool's maximum",
    )
    parser.add_argument("--n-strata", type=int, default=4)
    parser.add_argument("--periods", type=float, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--max-host-depth-ppm", type=float, default=3000.0)
    parser.add_argument("--sigma-clip", type=float, default=5.0)
    parser.add_argument("--window-length", type=int, default=401)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="keep the built shard set here instead of a temp dir, so a failed scoring pass "
        "leaves something to inspect and re-score by hand. The driver still rebuilds it "
        "on every run — see the note in main()",
    )
    parser.add_argument(
        "--also-routable-in",
        type=Path,
        default=None,
        help=(
            "restrict hosts to those also routable out-of-fold in this run directory. "
            "Required for a within-protocol comparison: the champion's fold membership "
            "predates K2 and the current catalogue, so the two runs cover different "
            "populations and an unrestricted pair would compare two of them again"
        ),
    )
    args = parser.parse_args()
    kind = run_kind(args.run_dir)
    log.info("[control-arm] %s is a %s run", args.run_dir.name, kind)
    # Resolved before the build loop, not inside it: this raised after 12.5
    # minutes of view building the first time a modern dual-view run was
    # scored, and the width is knowable from the bundle before any host is read.
    aux_dim = dualview_aux_dim(args.run_dir) if kind == "dualview" else 0
    if kind == "dualview":
        log.info("[control-arm] %s expects a %d-dim aux row", args.run_dir.name, aux_dim)

    labels_all = pd.read_parquet(args.labels)
    predictions = pd.read_parquet(args.run_dir / "predictions.parquet")
    # The champion's predictions carry `fold` and `y_true` but no `mission`, so
    # without this join the operating point would be fitted over Kepler+TESS
    # together — a threshold set partly by a mission with no serving stake.
    if "mission" not in predictions.columns:
        predictions = predictions.merge(
            labels_all[["tic_id", "mission"]].drop_duplicates("tic_id"),
            on="tic_id",
            how="left",
            validate="many_to_one",
        )
    if "label" not in predictions.columns:
        predictions["label"] = predictions["y_true"]
    if "score" not in predictions.columns:
        predictions["score"] = predictions["prob_calibrated"]
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

    labels = labels_all[labels_all["mission"] == "TESS"]
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
    shared_with = None
    if args.also_routable_in is not None:
        other = pd.read_parquet(args.also_routable_in / "predictions.parquet")
        other_folds = fold_assignment(other)
        before = len(routable)
        routable = routable[routable["tic_id"].isin(other_folds)]
        shared_with = str(args.also_routable_in)
        log.info(
            "[control-arm] %d routable in both runs (%d dropped by %s)",
            len(routable),
            before - len(routable),
            args.also_routable_in.name,
        )

    dumps_path = args.momentum_dumps or (Path("data") / "tables" / "momentum_dumps.parquet")
    if dumps_path.exists():
        dumps = pd.read_parquet(dumps_path)["time"].to_numpy(dtype=float)
        log.info("[control-arm] %d flagged cadences from %s", dumps.size, dumps_path)
    else:
        dumps = np.empty(0, dtype=float)
        log.info(
            "[control-arm] no momentum-dump table at %s — that branch reads absent, which is "
            "correct for a run that has no momentum branch and wrong for one that has",
            dumps_path,
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
        per_label_per_stratum=args.per_stratum or max(1, args.hosts // (2 * args.n_strata)),
        n_strata=args.n_strata,
        seed=args.seed,
    )
    log.info("[control-arm] %s", matched.report())
    if matched.n == 0:
        raise SystemExit("no matched hosts — relax --max-host-depth-ppm or --n-strata")

    view_sets, rows = [], []
    dualview_rows: list[dict] = []
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
                if kind == "branch":
                    views, scalars = build_host_views(
                        tic_id,
                        fits,
                        period,
                        args.sigma_clip,
                        args.window_length,
                        args.polyorder,
                        momentum_dumps=dumps,
                    )
                    view_sets.append(views)
                else:
                    gview, lview, aux = build_host_dualview(
                        fits,
                        period,
                        host,
                        args.sigma_clip,
                        args.window_length,
                        args.polyorder,
                        aux_dim=aux_dim,
                    )
                    scalars = {"tic_id": tic_id, "mission": "TESS", "period": period}
                    dualview_rows.append(
                        {"tic_id": tic_id, "global_view": gview, "local_view": lview, "aux": aux}
                    )
            except Exception as exc:
                log.warning("[control-arm] TIC %d P=%.1f failed: %s", tic_id, period, exc)
                continue
            scalars["label"] = int(host["label"])
            scalars["baseline_days"] = float(host["baseline_days"])
            scalars["stratum"] = int(host["stratum"])
            rows.append(scalars)
            log.info("[control-arm] built TIC %d P=%.1f d", tic_id, period)

    if not rows:
        raise SystemExit("no host views built")

    args.out.mkdir(parents=True, exist_ok=True)
    if kind == "branch":
        arrays = stack_view_sets(view_sets, control_scalars(rows))
        problems = arrays.validate()
        if problems:
            raise ValueError(f"control-arm view set is malformed: {problems}")
        # Persisted rather than a TemporaryDirectory when --shard-dir is given,
        # so a crash in the scoring pass leaves the ~35 min build on disk to
        # inspect and re-score by hand.
        #
        # It is deliberately **not** a resume: the build runs unconditionally
        # even when the directory already holds a shard set. Skipping it would
        # mean a directory left by a different host draw, seed or period list
        # gets scored as though it were this one — a wrong measurement that
        # reports a completely plausible pass rate, which is the failure mode
        # this project keeps paying for. Rebuilding costs 35 minutes; a silent
        # mismatch costs the result's credibility.
        stack = contextlib.ExitStack()
        with stack:
            if args.shard_dir is not None:
                shard_dir = args.shard_dir
            else:
                shard_dir = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "shards"
            write_viewset_shards(arrays, shard_dir)
            frame, scored = score_through_run(args.run_dir, shard_dir, folds)
    else:
        scored = score_through_dualview(args.run_dir, dualview_rows, folds)
        frame = pd.DataFrame(rows)
    frame = frame.copy()
    frame["score"] = scored
    unscored = int(frame["score"].isna().sum())
    if unscored:
        log.warning("[control-arm] %d row(s) unscored and dropped", unscored)
        frame = frame[frame["score"].notna()]

    results: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "run_kind": kind,
        "also_routable_in": shared_with,
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
                "not comparable with the live path's 26.4%; the champion must be "
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
