"""Build stage 8's synthetic-negative view set — arm N.

Draws real TESS hosts whose observation baselines match the **positives'**,
destroys any transit in their light curves, and emits a `ViewSetArrays` of rows
labelled negative. Merged into the training view set, those rows carry a label
that is correct regardless of how long the star was observed, which is what
breaks the catalogue's baseline-label association by construction.

    python pipeline/scripts/build_synthetic_negatives.py --n 400
    python pipeline/scripts/shard_viewset.py \
        --extra data/processed/synthetic_negatives \
        --out-dir data/processed/viewset_tfrecords_synneg

    clean -> flatten -> scramble|invert -> assert_transit_destroyed
          -> build_view_set at the host's own ephemeris

Three decisions this script makes, all of which could quietly ruin the arm.

**1. A synthetic row gets its own `tic_id` and keeps its parent's `group_tic`.**
A scrambled row and the row it came from are the same star seen twice. Grouped
CV splits on `group_tic`, so they cannot land in different folds — otherwise the
model is tested on a star whose noise, gaps and systematics it trained on. The
distinct `tic_id` keeps the split and weight tables one-to-one.

**2. DV scalars are INHERITED from the parent by default, not masked.** Masking
is the physically honest choice — no DV report describes a scrambled curve — and
it is a trap. Every synthetic negative would carry `dv_usable=False` while 93%
of real TESS rows carry True, handing the model a perfect shortcut: *mask off ⇒
negative*. It would learn that instead of the intervention, and the arm would
report a beautiful correlation drop while having taught the model nothing.
Inheriting instead means the row says "DV called this a strong detection and the
light curve has no transit in it", which is the lesson worth teaching. The
alternative is available as `--dv-policy mask` and the shortcut is guarded
either way — see `_assert_no_dv_shortcut`.

**3. The ephemeris is the host's own.** A synthetic negative at a *random*
ephemeris is a different and easier task: the model could reject it on phase
coverage alone. Folded at the ephemeris where the catalogue says a transit
should be, and finding nothing, is the discrimination that matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoplanet_hunter.datasets.synthetic_negatives import (
    INVERT,
    KINDS,
    SCRAMBLE,
    assert_transit_destroyed,
    draw_negative_hosts,
    make_synthetic_negative,
)
from exoplanet_hunter.datasets.viewset_io import stack_view_sets
from exoplanet_hunter.datasets.viewset_tfrecords import FEATURE_COLUMNS, MASK_COLUMNS
from exoplanet_hunter.preprocess import clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.preprocess.viewset import build_view_set
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

SIGMA_CLIP = 5.0
WINDOW_LENGTH = 401
POLYORDER = 2

#: Synthetic rows are numbered from here downwards, so a `tic_id` in the index
#: is either a real TIC or is obviously not one. A positive offset would collide
#: with a real TIC the day the catalogue grows past it.
SYNTHETIC_TIC_BASE = -1

#: How far the synthetic set's `dv_usable` rate may sit from the real set's
#: before the mask becomes a usable shortcut for the label.
MAX_DV_RATE_GAP = 0.10


def _assert_no_dv_shortcut(synthetic: pd.DataFrame, real: pd.DataFrame) -> float:
    """Raise if `dv_usable` alone separates synthetic negatives from real rows.

    The failure this prevents is the arm appearing to work while teaching the
    model something else entirely: if every synthetic negative is mask-off and
    almost every real row is mask-on, "mask off" *is* the label, the correlation
    with observation baseline duly collapses, and none of it came from the
    intervention.
    """
    if "dv_usable" not in synthetic.columns or "dv_usable" not in real.columns:
        raise KeyError("both frames need dv_usable to check for a mask shortcut")
    synth_rate = float(synthetic["dv_usable"].mean())
    real_rate = float(real["dv_usable"].mean())
    gap = abs(synth_rate - real_rate)
    if gap > MAX_DV_RATE_GAP:
        raise ValueError(
            f"dv_usable is True for {synth_rate:.1%} of synthetic negatives against "
            f"{real_rate:.1%} of real rows — a gap of {gap:.1%} (limit {MAX_DV_RATE_GAP:.0%}). "
            "The presence mask separates the classes on its own, so the model can learn "
            "'mask off means negative' and the arm would report a correlation drop it did "
            "not cause. Use --dv-policy inherit"
        )
    log.info(
        "[synthetic-negatives] dv_usable %.1f%% synthetic vs %.1f%% real (gap %.1f%%)",
        100 * synth_rate,
        100 * real_rate,
        100 * gap,
    )
    return gap


def build_one(
    host: dict, fits_path: Path, kind: str, seed: int, dv_policy: str
) -> tuple[Any, dict]:
    """One synthetic negative: views plus its scalar row, or raise."""
    import lightkurve as lk

    raw = lk.read(str(fits_path))
    cleaned = clean_lightcurve(raw, sigma_clip=SIGMA_CLIP)
    flat = flatten_lightcurve(cleaned, window_length=WINDOW_LENGTH, polyorder=POLYORDER)

    time = np.asarray(flat.time.value, dtype=float)
    original = np.asarray(flat.flux.value, dtype=float)
    period = float(host["period"])
    duration = float(host["duration"]) if np.isfinite(host.get("duration", np.nan)) else 0.1
    t0 = float(host["t0"]) if np.isfinite(host.get("t0", np.nan)) else float(np.nanmin(time))

    constructed = make_synthetic_negative(time, original, kind, seed=seed)
    # Inversion turns the dip into a brightening rather than removing it, so the
    # magnitude check would fail on a curve that is unambiguously not a transit.
    if kind == SCRAMBLE:
        assert_transit_destroyed(time, original, constructed, period, t0, duration)

    flat.flux[:] = constructed
    views = build_view_set(flat, period=period, t0=t0, duration=duration, raw_lc=raw)

    observed = int(views.observed_transit_count)
    expected = int(views.expected_transit_count)
    scalars: dict[str, Any] = {
        "mission": "TESS",
        "label": 0,
        "period": period,
        "observed_transit_count": observed,
        "expected_transit_count": expected,
        "transit_completeness": observed / expected if expected else 0.0,
        "secondary_phase": float(views.secondary_phase),
        "synthetic_kind": kind,
    }
    if dv_policy == "inherit":
        for column in (*FEATURE_COLUMNS, *MASK_COLUMNS):
            if column not in scalars and column in host:
                scalars[column] = host[column]
    return views, scalars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scalars", type=Path, default=Path("data/processed/viewset_scalars.parquet")
    )
    parser.add_argument("--labels", type=Path, default=Path("data/labels/labels.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/tess/lightcurves"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/synthetic_negatives"),
        help="directory, written as a ViewSetArrays (viewset.npz + viewset_scalars.parquet)",
    )
    parser.add_argument("--n", type=int, default=400, help="synthetic negatives to build")
    parser.add_argument("--n-strata", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dv-policy",
        choices=("inherit", "mask"),
        default="inherit",
        help="inherit the parent's DV scalars (default) or write them unmeasured. "
        "'mask' is physically honest and hands the model a mask-off shortcut for the label",
    )
    args = parser.parse_args()

    scalars = pd.read_parquet(args.scalars)
    labels = pd.read_parquet(args.labels)
    real = scalars[scalars["mission"] == "TESS"].copy()
    # The ephemeris the fold is built at lives in labels.parquet, not the view
    # set index — the index carries `period` but neither `t0` nor `duration`.
    pool = real.merge(
        labels[["tic_id", "t0", "duration"]].drop_duplicates("tic_id"), on="tic_id", how="left"
    )
    cached = {int(p.stem.split("_")[1]) for p in args.raw.glob("tic_*.fits")}
    pool = pool[pool["tic_id"].isin(cached)]
    log.info("[synthetic-negatives] %d cached TESS hosts eligible", len(pool))

    draw = draw_negative_hosts(pool, n=args.n, seed=args.seed, n_strata=args.n_strata)

    view_sets, rows = [], []
    for i, host in enumerate(draw.hosts.to_dict("records")):
        tic_id = int(host["tic_id"])
        fits = next(args.raw.glob(f"tic_{tic_id}_*.fits"), None) or next(
            args.raw.glob(f"tic_{tic_id}.fits"), None
        )
        if fits is None:
            log.warning("[synthetic-negatives] TIC %d has no cached FITS — skipped", tic_id)
            continue
        # Alternated rather than drawn at random, so the two constructions are
        # balanced by construction and a later split on `synthetic_kind` compares
        # equal-sized halves.
        kind = KINDS[i % len(KINDS)]
        try:
            views, scalar_row = build_one(host, fits, kind, args.seed + i, args.dv_policy)
        except Exception as exc:
            log.warning("[synthetic-negatives] TIC %d (%s) failed: %s", tic_id, kind, exc)
            continue
        scalar_row["tic_id"] = SYNTHETIC_TIC_BASE - i
        scalar_row["group_tic"] = tic_id
        view_sets.append(views)
        rows.append(scalar_row)
        log.info("[synthetic-negatives] built %s from TIC %d", kind, tic_id)

    if not rows:
        raise SystemExit("no synthetic negatives built")

    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in MASK_COLUMNS:
        if column not in frame.columns:
            frame[column] = False
    _assert_no_dv_shortcut(frame, real)

    arrays = stack_view_sets(view_sets, frame)
    problems = arrays.validate()
    if problems:
        raise ValueError(f"synthetic-negative view set is malformed: {problems}")

    # Through `ViewSetArrays.save`, not a bespoke npz: `shard_viewset.py --extra`
    # loads it with `ViewSetArrays.load`, and a second serialisation format for
    # the same object is how two readers drift apart.
    arrays.save(args.out)
    by_kind = frame["synthetic_kind"].value_counts().to_dict()
    log.info(
        "[synthetic-negatives] wrote %d rows (%d scramble / %d invert) to %s",
        len(frame),
        by_kind.get(SCRAMBLE, 0),
        by_kind.get(INVERT, 0),
        args.out,
    )


if __name__ == "__main__":
    main()
