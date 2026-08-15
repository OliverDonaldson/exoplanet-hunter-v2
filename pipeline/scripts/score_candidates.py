"""Bulk-score held-out planet candidates with the promoted 5-fold ensemble.

Scores every row of ``data/tables/labels/candidates.parquet`` through the same
preprocessing and aux layout the API serves (``features/aux.py``), in whatever
width the registered bundle was trained on, and reports a richer uncertainty
breakdown than a single /score call: ensemble mean/std, p10/p90, between-fold
disagreement and within-fold MC-Dropout spread. Output is
``results/candidates_scored.parquet`` (resumable, atomic).

Usage:
    # Full pool (both missions, all 6,200):
    python scripts/score_candidates.py

    # Sanity-check on N=10:
    python scripts/score_candidates.py max_candidates=10

    # TESS only:
    python scripts/score_candidates.py limit_mission=TESS

    # Override fold-model bundle (defaults to the promoted registry run):
    python scripts/score_candidates.py cv_dir=models/cv/<HASH>
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from tqdm.auto import tqdm

from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.features.aux import build_aux_row
from exoplanet_hunter.features.centroid import extract_centroid_offset
from exoplanet_hunter.features.noise import pink_noise_snr
from exoplanet_hunter.models.uncertainty import mc_dropout_predict
from exoplanet_hunter.preprocess import build_views, clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.scoring.diagnostics import (
    odd_even_depths,
    significant_secondary,
    unphysical_duration,
)
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)

DEFAULT_OUT = "results/candidates_scored.parquet"

#: Concurrent MAST fetches while warming the FITS cache before scoring.
#: Modest to stay polite to the archive (Kepler targets also fan out
#: internally over quarter files).
_PREFETCH_WORKERS = 4


def _registry_cv_dir(root: Path) -> Path:
    """Default to the promoted run, exactly like the serving path."""
    registry = json.loads((root / "models" / "registry.json").read_text())
    return root / "models" / "cv" / registry["run_id"]


def _load_fold_bundles(cv_dir: Path, n_folds: int = 5) -> list[dict]:
    bundles = []
    for k in range(n_folds):
        fold_dir = cv_dir / f"fold_{k}"
        model = tf.keras.models.load_model(str(fold_dir / "cnn_dualview.keras"), compile=False)
        bundle = joblib.load(fold_dir / "cnn_calibrator.joblib")
        bundles.append(
            {
                "k": k,
                "model": model,
                "calibrator": bundle.get("calibrator"),
                "aux_pipeline": bundle.get("aux_pipeline"),
                "aux_dim": bundle.get("aux_dim"),
                "threshold": float(bundle.get("threshold", 0.5)),
            }
        )
    return bundles


def _aux_vector(
    row: dict,
    raw_lc,
    flat_time: np.ndarray,
    flat_flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    mission: str,
    aux_dim: int,
) -> tuple[np.ndarray, float]:
    """Aux vector in the served bundle's layout — see features/aux.py.

    Stellar params come from the candidate row for every mission (an EPIC id
    is not a TIC id). Returns (aux_vector, centroid_snr) so the caller can
    record centroid_snr.
    """
    try:
        centroid_snr = float(extract_centroid_offset(raw_lc, period, t0, duration))
    except Exception as exc:
        log.debug("centroid extract failed tic=%s: %s", row.get("tic_id"), exc)
        centroid_snr = float("nan")

    extra: dict[str, object]
    if aux_dim >= 13:
        pn = pink_noise_snr(flat_time, flat_flux, period, t0, duration)
        oe = odd_even_depths(flat_time, flat_flux, period, t0, duration)
        sec = significant_secondary(flat_time, flat_flux, period, t0, duration)
        dc = unphysical_duration(
            period, duration, stellar_radius=row.get("radius"), stellar_logg=row.get("logg")
        )
        extra = {
            "pink_snr": pn.snr if pn is not None else None,
            "oe_depth_sigma": oe.depth_diff_sigma if oe is not None else None,
            "oe_timing_sigma": oe.timing_diff_sigma if oe is not None else None,
            "secondary_sig": sec.secondary_significance if sec is not None else None,
            "q_ratio": dc.q_ratio if dc is not None else None,
        }
    else:
        # Matches training: catalogue SNR exists for Kepler, not for TESS.
        extra = {"catalogue_snr": row.get("snr") if mission == "Kepler" else None}

    aux = build_aux_row(
        aux_dim,
        teff=row.get("teff"),
        radius=row.get("radius"),
        logg=row.get("logg"),
        tmag=row.get("tmag"),
        depth=row.get("depth"),
        duration=duration,
        period=period,
        centroid_snr=centroid_snr,
        **extra,  # type: ignore[arg-type]
    )
    return aux, centroid_snr


def _score_one(
    row: dict, folds: list[dict], downloader: LightCurveDownloader, cfg: DictConfig, n_mc: int
) -> dict:
    tic = int(row["tic_id"])
    mission = str(row.get("mission", "TESS"))
    period = float(row["period"])
    t0 = float(row["t0"])
    duration = float(row["duration"])

    base = {
        "candidate_idx": row["candidate_idx"],
        "tic_id": tic,
        "toi": row.get("toi"),
        "name": row.get("name"),
        "disposition": row.get("disposition"),
        "mission": mission,
        "period": period,
        "t0": t0,
        "duration": duration,
        "depth": row.get("depth"),
        "scored_at": pd.Timestamp.utcnow().isoformat(),
    }

    res = downloader.download_one(tic, mission=mission)
    if not res.success or res.path is None:
        return {**base, "status": "no_fits", "reason": res.reason}

    import lightkurve as lk

    try:
        raw = lk.read(str(res.path))
    except Exception as exc:
        return {**base, "status": "read_fail", "reason": str(exc)}

    try:
        cleaned = clean_lightcurve(raw, sigma_clip=float(cfg.preprocess.cleaning.sigma_clip))
        flat = flatten_lightcurve(
            cleaned,
            window_length=int(cfg.preprocess.flatten.window_length),
            polyorder=int(cfg.preprocess.flatten.polyorder),
            period=period,
            t0=t0,
            duration=duration,
        )
        views = build_views(
            flat,
            period=period,
            t0=t0,
            duration=duration,
            global_bins=int(cfg.preprocess.views.global_bins),
            local_bins=int(cfg.preprocess.views.local_bins),
            local_durations=float(cfg.preprocess.views.local_durations),
        )
    except Exception as exc:
        return {**base, "status": "preprocess_fail", "reason": str(exc)}

    aux_dim = next((f["aux_dim"] for f in folds if f["aux_dim"]), 9)
    aux_raw, centroid_snr = _aux_vector(
        row,
        raw,
        np.asarray(flat.time.value, dtype=float),
        np.asarray(flat.flux.value, dtype=float),
        period,
        t0,
        duration,
        mission,
        aux_dim,
    )

    inputs_static = {
        "global_view": views.global_view[None, :, None].astype(np.float32),
        "local_view": views.local_view[None, :, None].astype(np.float32),
    }

    fold_means: list[float] = []
    fold_stds: list[float] = []
    all_calibrated: list[np.ndarray] = []

    for f in folds:
        aux_scaled = f["aux_pipeline"].transform(aux_raw[None, :]).astype(np.float32)
        inputs = {**inputs_static, "aux_features": aux_scaled}
        mc = mc_dropout_predict(f["model"], inputs, n_samples=n_mc)
        # mc.samples is shape (n_mc,) for batch=1
        uncal = np.atleast_1d(mc.samples).astype(np.float64)
        cal = np.asarray(f["calibrator"].predict(uncal)) if f["calibrator"] is not None else uncal
        all_calibrated.append(cal)
        fold_means.append(float(cal.mean()))
        fold_stds.append(float(cal.std()))

    samples = np.concatenate(all_calibrated)
    return {
        **base,
        "status": "ok",
        "centroid_snr": float(centroid_snr) if np.isfinite(centroid_snr) else None,
        "prob_mean": float(samples.mean()),
        "prob_std": float(samples.std()),
        "prob_p10": float(np.percentile(samples, 10)),
        "prob_p90": float(np.percentile(samples, 90)),
        "fold_disagree": float(np.std(fold_means)),  # between-fold spread
        "mc_disagree": float(np.mean(fold_stds)),  # within-fold dropout
        "fold_means": fold_means,
    }


def _checkpoint(rows: list[dict], out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out_path)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    n_mc = int(getattr(cfg, "n_mc", 30))
    max_candidates = getattr(cfg, "max_candidates", None)
    limit_mission = getattr(cfg, "limit_mission", None)
    cv_dir_arg = getattr(cfg, "cv_dir", None)
    out_arg = str(getattr(cfg, "out_path", DEFAULT_OUT))
    save_every = int(getattr(cfg, "save_every", 25))

    if cv_dir_arg is None:
        cv_dir = _registry_cv_dir(paths.root)
    else:
        cv_dir = (
            Path(cv_dir_arg) if Path(cv_dir_arg).is_absolute() else paths.root / str(cv_dir_arg)
        )
    out_path = Path(out_arg) if Path(out_arg).is_absolute() else paths.root / out_arg
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("[score-candidates] cv_dir=%s", cv_dir)
    log.info("[score-candidates] out=%s  n_mc=%d  save_every=%d", out_path, n_mc, save_every)

    # ----- candidates ---------------------------------------------------------
    candidates = pd.read_parquet(paths.data_labels / "candidates.parquet").reset_index(drop=True)
    candidates["candidate_idx"] = candidates.index.astype(int)
    log.info("[score-candidates] loaded %d candidates", len(candidates))

    if limit_mission:
        candidates = candidates[candidates.mission == limit_mission].reset_index(drop=True)
        log.info("[score-candidates] filtered to mission=%s → %d", limit_mission, len(candidates))
    if max_candidates is not None:
        candidates = candidates.head(int(max_candidates)).reset_index(drop=True)
        log.info("[score-candidates] truncated to first %d", len(candidates))

    # ----- resume from prior run --------------------------------------------
    done_ids: set[int] = set()
    prior_rows: list[dict] = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        prior_rows = prev.to_dict("records")
        done_ids = {int(r["candidate_idx"]) for r in prior_rows if pd.notna(r.get("candidate_idx"))}
        log.info("[score-candidates] resuming with %d prior rows", len(done_ids))

    todo = candidates[~candidates.candidate_idx.isin(done_ids)].reset_index(drop=True)
    log.info("[score-candidates] %d rows to score this run", len(todo))
    if len(todo) == 0:
        log.info("[score-candidates] nothing to do")
        return

    # ----- model bundles ----------------------------------------------------
    folds = _load_fold_bundles(cv_dir, n_folds=5)
    log.info("[score-candidates] loaded %d fold bundles", len(folds))

    # ----- downloaders ------------------------------------------------------
    # TESS PCs cache to data/raw/ (alongside training cache, distinct IDs).
    # Kepler PCs cache to data/raw/kepler/lightcurves/ (or whatever KEPLER_RAW_DIR resolved to).
    tess_dl = LightCurveDownloader(paths.data_raw, author="SPOC", cadence=120)
    kepler_dl = LightCurveDownloader(
        paths.data_raw,
        kepler_cache_dir=paths.data_raw_kepler,
        k2_cache_dir=paths.data_raw_k2,
        author="Kepler",
        cadence=None,
    )

    todo_records = todo.to_dict("records")

    # ----- prefetch downloads in parallel -----------------------------------
    # FITS fetches (30 s-3 min each of MAST latency) are the wall-time killer;
    # the scoring loop below is TF-bound and already saturates the CPU. Warm
    # the cache up front with a few concurrent download threads so the loop
    # then runs ~100% cache-hit. Split by mission — TESS and Kepler use
    # different downloader instances and cache dirs. Blocks until done, so no
    # download runs concurrently with scoring.
    tess_ids = [int(r["tic_id"]) for r in todo_records if str(r.get("mission", "TESS")) != "Kepler"]
    kepler_ids = [
        int(r["tic_id"]) for r in todo_records if str(r.get("mission", "TESS")) == "Kepler"
    ]
    if tess_ids:
        log.info("[score-candidates] prefetching %d TESS light curves", len(tess_ids))
        tess_dl.download_many(tess_ids, workers=_PREFETCH_WORKERS)
    if kepler_ids:
        log.info("[score-candidates] prefetching %d Kepler light curves", len(kepler_ids))
        kepler_dl.download_many(
            kepler_ids, missions=["Kepler"] * len(kepler_ids), workers=_PREFETCH_WORKERS
        )

    # ----- scoring loop -----------------------------------------------------
    # Sequential by design: each _score_one runs 5 folds x MC-Dropout, which
    # saturates the CPU (Keras predict is not guaranteed thread-safe), and the
    # slow part — the download — was just parallelised above.
    out_rows = list(prior_rows)
    t_start = time.time()

    for i, row_dict in enumerate(tqdm(todo_records, desc="scoring")):
        mission = str(row_dict.get("mission", "TESS"))
        dl = kepler_dl if mission == "Kepler" else tess_dl
        try:
            result = _score_one(row_dict, folds, dl, cfg, n_mc=n_mc)
        except Exception as exc:
            log.exception(
                "unexpected error idx=%s tic=%s",
                row_dict.get("candidate_idx"),
                row_dict.get("tic_id"),
            )
            result = {
                "candidate_idx": row_dict["candidate_idx"],
                "tic_id": int(row_dict["tic_id"]),
                "mission": mission,
                "status": "error",
                "reason": str(exc),
            }
        out_rows.append(result)

        if (i + 1) % save_every == 0:
            _checkpoint(out_rows, out_path)
            elapsed = time.time() - t_start
            log.info(
                "[score-candidates] checkpoint %d/%d  elapsed=%.0fs  rate=%.1fs/row",
                i + 1,
                len(todo),
                elapsed,
                elapsed / max(1, i + 1),
            )

    _checkpoint(out_rows, out_path)
    elapsed = time.time() - t_start
    scored = pd.DataFrame(out_rows)
    n_ok = int((scored.status == "ok").sum())
    log.info(
        "[score-candidates] DONE  %d total rows  ok=%d  elapsed=%.0fs → %s",
        len(out_rows),
        n_ok,
        elapsed,
        out_path,
    )
    if n_ok > 0:
        ok = scored[scored.status == "ok"]
        log.info(
            "[score-candidates] prob_mean: min=%.3f median=%.3f max=%.3f   high-conf (>0.9): %d",
            ok.prob_mean.min(),
            ok.prob_mean.median(),
            ok.prob_mean.max(),
            int((ok.prob_mean > 0.9).sum()),
        )


if __name__ == "__main__":
    main()
