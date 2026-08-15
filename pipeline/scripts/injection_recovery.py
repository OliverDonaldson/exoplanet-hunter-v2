"""Injection-recovery completeness for the served model.

Injects synthetic box transits of known depth into real light curves, scores
them through the live path (`TargetScorer.score(inject=...)`, so preprocess,
aux layout and ensemble are whatever the registry currently serves), and tallies
the recovered fraction per transit S/N bin.

This is the sensitivity statement CV ROC-AUC cannot make: "above S/N ~N the
served model passes >X% of genuine transits at its own threshold."

The injected ephemeris is supplied to the scorer, so this measures the
*classifier*, not a period search. Hosts are real TESS targets, so any signal
already in the light curve is folded at the wrong period and smears into the
noise; hosts with a deep known transit are skipped, and injected periods near
the host's own period (or its 2:1 harmonics) are dropped.

Each injection targets a transit S/N and the depth is solved per host, so the
curve is sampled where it turns over rather than piling into one bin (host
noise spans orders of magnitude).

Usage (from the repository root):

    python pipeline/scripts/injection_recovery.py --hosts 40
    python pipeline/scripts/injection_recovery.py --hosts 3 --periods 5.0 --snr-grid 3 7 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_hunter.eval.control_arm import baseline_matched_hosts
from exoplanet_hunter.eval.injection_recovery import (
    completeness_curve,
    count_transits,
    noise_ppm,
    transit_snr,
)
from exoplanet_hunter.eval.observation_bias import BASELINE_DAYS, baseline_days
from exoplanet_hunter.preprocess import clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.scoring import NoLightCurveError, TargetScorer
from exoplanet_hunter.scoring.service import InjectionSpec
from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

DEFAULT_PERIODS = (3.0, 7.0, 12.0)
DEFAULT_SNR_GRID = (3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0)
#: Central transit of a Sun-like star, scaled from Earth's 13 h (Seager &
#: Mallen-Ornelas 2003) — an injection needs *a* duration, not a fitted one.
SOLAR_DURATION_HOURS = 13.0
#: Deeper than this is an eclipsing binary, not a planet — skip rather than
#: inject something the model should reject anyway.
MAX_INJECT_DEPTH_PPM = 200_000.0


def transit_duration_hours(period_days: float) -> float:
    return SOLAR_DURATION_HOURS * (period_days / 365.25) ** (1.0 / 3.0)


def depth_for_snr(snr: float, cdpp_ppm: float, n_transits: int) -> float:
    """Invert transit_snr: the depth this host needs to land at `snr`.

    Hosts differ in brightness and coverage by orders of magnitude, so a fixed
    depth grid piles every injection into one or two S/N bins. Targeting S/N
    instead samples the curve where it actually turns over.
    """
    return snr * cdpp_ppm / np.sqrt(n_transits)


def snr_bin_edges(levels: np.ndarray) -> np.ndarray:
    """One bin per requested S/N level, split at geometric midpoints."""
    levels = np.sort(np.asarray(levels, dtype=float))
    if levels.size == 1:
        return np.array([levels[0] * 0.5, levels[0] * 2.0])
    mids = np.sqrt(levels[:-1] * levels[1:])
    return np.concatenate([[levels[0] ** 2 / mids[0]], mids, [levels[-1] ** 2 / mids[-1]]])


def _harmonic_of(period: float, host_period: float, tol: float = 0.02) -> bool:
    """True if `period` sits within `tol` of the host's period or its 2:1 harmonics."""
    if not np.isfinite(host_period) or host_period <= 0:
        return False
    return any(abs(period - host_period * r) / (host_period * r) < tol for r in (0.5, 1.0, 2.0))


def join_baseline(hosts: pd.DataFrame, viewset_index_path: Path) -> pd.DataFrame:
    """Attach `baseline_days` from the viewset scalars.

    **`labels.parquet` does not carry `expected_transit_count`** — it holds 18
    columns and that is not one of them — so the baseline the observation-bias
    work is defined against cannot be derived from the labels alone. The viewset
    index does carry it, keyed on `tic_id`.

    `validate="one_to_one"` rather than a bare merge: a repeated `tic_id` on
    either side would multiply host rows and every downstream rate would be
    computed over a population that does not exist. Five merges in
    `eval/comparison.py` had exactly that hole until 2026-08-08.
    """
    index = pd.read_parquet(viewset_index_path)
    needed = {"tic_id", "expected_transit_count", "period"}
    missing = needed - set(index.columns)
    if missing:
        raise KeyError(f"{viewset_index_path} is missing {sorted(missing)}")
    scalars = index[sorted(needed)].drop_duplicates("tic_id")
    joined = hosts.drop(columns=["expected_transit_count"], errors="ignore").merge(
        scalars.rename(columns={"period": "viewset_period"}),
        on="tic_id",
        how="inner",
        validate="one_to_one",
    )
    # The viewset period is the one `expected_transit_count` was counted
    # against, so the baseline is derived from that pair and not from the
    # catalogue period, which can differ after a refresh.
    joined[BASELINE_DAYS] = baseline_days(
        joined.rename(columns={"viewset_period": "period"})[["expected_transit_count", "period"]]
    )
    return joined


def select_hosts(
    labels_path: Path,
    raw_dir: Path,
    *,
    n_hosts: int,
    max_host_depth_ppm: float,
    allow_download: bool,
    seed: int,
    host_label: int | None = None,
    viewset_index_path: Path | None = None,
    match_baseline: bool = False,
    n_strata: int = 4,
) -> pd.DataFrame:
    """TESS hosts shallow enough that their own signal will not dominate.

    With `match_baseline`, planet and false-positive hosts are drawn matched on
    observation baseline — the harness the roadmap has owed since old 2(b).
    Unmatched, the control arm's 46.7 / 12.3 split is confounded by the +0.387
    correlation between baseline and the label on TESS, so a difference between
    the two host populations cannot be attributed to the model at all.

    Matching needs a `viewset_index_path` because the baseline is not derivable
    from `labels.parquet`; asking for it without one raises rather than falling
    back to an unmatched draw that reports the same column names.
    """
    df = pd.read_parquet(labels_path)
    df = df[df["mission"] == "TESS"]
    if host_label is not None:
        df = df[df["label"] == host_label]
    df = df[df["depth"].fillna(np.inf) * 1e6 <= max_host_depth_ppm]
    if not allow_download:
        cached = {int(p.stem.split("_")[1]) for p in raw_dir.glob("tic_*.fits")}
        df = df[df["tic_id"].isin(cached)]
    df = df.drop_duplicates("tic_id")
    if df.empty:
        return df

    if not match_baseline:
        if viewset_index_path is not None:
            df = join_baseline(df, viewset_index_path)  # reported, not matched on
        return df.sample(n=min(n_hosts, len(df)), random_state=seed)

    if viewset_index_path is None:
        raise ValueError(
            "match_baseline needs viewset_index_path: labels.parquet carries no "
            "expected_transit_count, so baseline_days cannot be derived from it"
        )
    if host_label is not None:
        raise ValueError(
            "match_baseline draws both labels by construction; --host-label would "
            "leave every stratum single-label and drop the whole draw"
        )
    matched = baseline_matched_hosts(
        join_baseline(df, viewset_index_path),
        per_label_per_stratum=max(1, n_hosts // (2 * n_strata)),
        n_strata=n_strata,
        seed=seed,
    )
    log.info("[injection] baseline-matched hosts: %s", matched.report())
    return matched.hosts


def host_baseline(scorer: TargetScorer, tic_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (time, flux) of the un-injected host — the noise reference."""
    import lightkurve as lk

    res = scorer.downloader.download_one(tic_id)
    if not res.success or res.path is None:
        raise NoLightCurveError(f"no SPOC light curve for TIC {tic_id} ({res.reason})")
    cleaned = clean_lightcurve(lk.read(str(res.path)), sigma_clip=scorer.preprocess.sigma_clip)
    flat = flatten_lightcurve(
        cleaned,
        window_length=scorer.preprocess.window_length,
        polyorder=scorer.preprocess.polyorder,
    )
    return (
        np.asarray(flat.time.value, dtype=float),
        np.asarray(flat.flux.value, dtype=float),
    )


def report(
    results: pd.DataFrame, edges: np.ndarray, levels: np.ndarray | None = None
) -> tuple[pd.DataFrame, float]:
    """Recovered fraction per S/N bin, plus the control-arm baseline.

    Hosts are dispositioned targets, not blank sky, so a run with no injected
    signal still passes some fraction — the model reads the host's own transit
    and stellar aux. The control arm (snr_target == 0, depth 0) measures that
    directly; `completeness_corrected` rescales the raw fraction onto it, so
    0 means "no better than the host alone".
    """
    scored = results.dropna(subset=["snr"])
    controls = scored[scored["snr_target"] == 0]
    baseline = float(controls["recovered"].mean()) if len(controls) else 0.0

    graded = scored[scored["snr_target"] > 0]
    centers, fraction, count = completeness_curve(
        graded["snr"].to_numpy(), graded["recovered"].to_numpy(), edges
    )
    if levels is not None and len(levels) == len(centers):
        centers = np.sort(np.asarray(levels, dtype=float))
    corrected = (
        np.clip((fraction - baseline) / (1.0 - baseline), 0.0, 1.0) if baseline < 1 else fraction
    )
    return (
        pd.DataFrame(
            {
                "snr": centers,
                "completeness": fraction,
                "completeness_corrected": corrected,
                "n": count,
            }
        ),
        baseline,
    )


def plot_completeness(
    curve: pd.DataFrame, out: Path, run_id: str, threshold: float, baseline: float = 0.0
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = curve["n"] > 0
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(
        curve.loc[ok, "snr"],
        curve.loc[ok, "completeness"],
        "o-",
        color="royalblue",
        label="raw",
    )
    if baseline > 0:
        ax.plot(
            curve.loc[ok, "snr"],
            curve.loc[ok, "completeness_corrected"],
            "s--",
            color="darkorange",
            label="baseline-corrected",
        )
        ax.axhline(
            baseline,
            color="crimson",
            lw=0.9,
            ls=":",
            label=f"control arm, no injection ({baseline:.2f})",
        )
        ax.legend(fontsize=8, loc="lower right")
    for x, y, n in curve.loc[ok, ["snr", "completeness", "n"]].itertuples(index=False):
        ax.annotate(
            f"{int(n)}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 7),
            fontsize=7,
            ha="center",
            color="0.45",
        )
    ax.axhline(0.5, color="0.8", lw=0.8, ls="--")
    ax.set_xlabel("injected transit S/N   (depth / CDPP · √n_transits)")
    ax.set_ylabel(f"fraction recovered  (prob ≥ {threshold:.2f})")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Injection-recovery completeness — served run {run_id[:8]}", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info("[injection] wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/labels.parquet"))
    parser.add_argument("--catalogue", type=Path, default=Path("data/catalogue/candidates.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/tess/lightcurves"))
    parser.add_argument("--out", type=Path, default=Path("results/injection_recovery.parquet"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/completeness.png"))
    parser.add_argument("--hosts", type=int, default=40, help="number of host light curves")
    parser.add_argument("--periods", type=float, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument(
        "--snr-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_SNR_GRID),
        help="target transit S/N per injection; the depth is solved per host",
    )
    parser.add_argument("--max-host-depth-ppm", type=float, default=3000.0)
    parser.add_argument("--allow-download", action="store_true", help="hosts need not be cached")
    parser.add_argument(
        "--host-label",
        type=int,
        choices=[0, 1],
        default=None,
        help="restrict hosts to false positives (0) or planet hosts (1)",
    )
    parser.add_argument(
        "--no-controls",
        dest="controls",
        action="store_false",
        help="skip the depth-0 control arm that measures the host-only pass rate",
    )
    parser.add_argument("--n-mc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hosts = select_hosts(
        args.labels,
        args.raw,
        n_hosts=args.hosts,
        max_host_depth_ppm=args.max_host_depth_ppm,
        allow_download=args.allow_download,
        seed=args.seed,
        host_label=args.host_label,
    )
    if hosts.empty:
        raise SystemExit("no eligible hosts — relax --max-host-depth-ppm or pass --allow-download")

    scorer = TargetScorer(
        models_dir=Path("models"), data_raw=args.raw, candidates_path=args.catalogue
    )
    run_id = scorer.ensemble.run_id
    levels_with_controls = ([0.0] if args.controls else []) + list(args.snr_grid)
    n_planned = len(hosts) * len(args.periods) * len(levels_with_controls)
    log.info(
        "[injection] %d hosts x %d periods x %d levels%s = %d injections, run %s (aux_dim %s)",
        len(hosts),
        len(args.periods),
        len(levels_with_controls),
        " (incl. control)" if args.controls else "",
        n_planned,
        run_id[:8],
        scorer.ensemble.aux_dim,
    )

    done = pd.read_parquet(args.out) if args.out.exists() else None
    seen: set[tuple[int, float, float]] = (
        set()
        if done is None
        else set(map(tuple, done[["tic_id", "period_days", "snr_target"]].to_numpy()))
    )
    rows: list[dict] = [] if done is None else done.to_dict("records")

    rng = np.random.default_rng(args.seed)
    n = 0
    for host in hosts.itertuples():
        tic_id = int(host.tic_id)
        try:
            base_time, base_flux = host_baseline(scorer, tic_id)
        except Exception as exc:
            log.warning("[injection] TIC %d skipped: %s", tic_id, exc)
            continue
        t_start = float(np.nanmin(base_time))

        for period in args.periods:
            if _harmonic_of(period, float(getattr(host, "period", np.nan))):
                log.info("[injection] TIC %d P=%.1f d skipped (host harmonic)", tic_id, period)
                continue
            duration_h = transit_duration_hours(period)
            duration_d = duration_h / 24.0
            t0 = t_start + float(rng.uniform(0.0, period))
            cdpp = noise_ppm(base_time, base_flux, duration_d)
            n_tr = count_transits(base_time, period, t0, duration_d)
            if not cdpp or n_tr < 2:
                log.info(
                    "[injection] TIC %d P=%.1f d skipped (cdpp=%s, n_transits=%d)",
                    tic_id,
                    period,
                    cdpp,
                    n_tr,
                )
                continue

            for snr_target in levels_with_controls:
                if (tic_id, period, snr_target) in seen:
                    continue
                depth_ppm = depth_for_snr(snr_target, cdpp, n_tr)
                if depth_ppm > MAX_INJECT_DEPTH_PPM:
                    log.info(
                        "[injection] TIC %d P=%.1f S/N=%.0f needs %.0f ppm — too deep, skipped",
                        tic_id,
                        period,
                        snr_target,
                        depth_ppm,
                    )
                    continue
                try:
                    outcome = scorer.score(
                        tic_id,
                        period_days=period,
                        t0_btjd=t0,
                        duration_hours=duration_h,
                        n_mc=args.n_mc,
                        inject=InjectionSpec(
                            period_days=period,
                            t0_btjd=t0,
                            duration_hours=duration_h,
                            depth=depth_ppm / 1e6,
                        ),
                    )
                except Exception as exc:
                    log.warning(
                        "[injection] TIC %d P=%.1f S/N=%.0f failed: %s",
                        tic_id,
                        period,
                        snr_target,
                        exc,
                    )
                    continue
                n += 1
                rows.append(
                    {
                        "tic_id": tic_id,
                        "host_label": int(host.label),
                        "period_days": float(period),
                        "snr_target": float(snr_target),
                        "depth_ppm": float(depth_ppm),
                        "duration_hours": duration_h,
                        "t0_btjd": t0,
                        "cdpp_ppm": float(cdpp),
                        "n_transits": int(n_tr),
                        "snr": transit_snr(depth_ppm, cdpp, n_tr),
                        "prob": outcome.prob_calibrated,
                        "prob_std": outcome.prob_std,
                        "threshold": outcome.threshold,
                        "recovered": bool(outcome.prob_calibrated >= outcome.threshold),
                        "run_id": run_id,
                    }
                )
                frame = pd.DataFrame(rows)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(args.out, index=False)  # checkpoint every injection
                log.info(
                    "[injection] %d/%d TIC %d P=%.1f d S/N=%.0f (%.0f ppm) -> %.3f %s",
                    n,
                    n_planned,
                    tic_id,
                    period,
                    snr_target,
                    depth_ppm,
                    outcome.prob_calibrated,
                    "RECOVERED" if rows[-1]["recovered"] else "missed",
                )

    results = pd.read_parquet(args.out)
    levels = np.asarray(args.snr_grid, dtype=float)
    curve, baseline = report(results, snr_bin_edges(levels), levels)
    threshold = float(results["threshold"].median())
    log.info("[injection] completeness by S/N bin:\n%s", curve.to_string(index=False))
    log.info("[injection] control arm (no injection) passes %.3f of the time", baseline)
    plot_completeness(curve, args.figure, run_id, threshold, baseline)


if __name__ == "__main__":
    main()
