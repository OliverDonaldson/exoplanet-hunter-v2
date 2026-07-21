"""End-to-end dataset build script.

Three stages:
  1. Build / refresh the labelled catalogue.
  2. Download light curves for every TIC.
  3. Clean, flatten, fold, and extract global+local views into a single
     `data/processed/views.npz`.

Idempotent — safe to re-run; downloads + processed views are cached.

Hydra entry point. Usage:

    python scripts/build_dataset.py                  # full dataset
    python scripts/build_dataset.py data=small       # tiny smoke set
"""

from __future__ import annotations

import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from exoplanet_hunter.data.catalog import CatalogRequest, build_label_catalog
from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.data.exofop import enrich_catalog_snr
from exoplanet_hunter.data.stellar import fetch_stellar_params
from exoplanet_hunter.features.centroid import extract_centroid_offset
from exoplanet_hunter.features.noise import pink_noise_snr
from exoplanet_hunter.preprocess import build_views, clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.scoring.diagnostics import (
    odd_even_depths,
    significant_secondary,
    unphysical_duration,
)
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    import os

    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    # Loud warning if Kepler targets are requested but KEPLER_RAW_DIR is unset.
    # Without it, ~75 GB of Kepler downloads will land on the internal SSD
    # instead of the SANDISK USB. Continues after the warning so a fresh
    # machine without the env var can still run end-to-end.
    n_kepler_total = int(cfg.data.get("n_confirmed_kepler", 0)) + int(
        cfg.data.get("n_false_pos_kepler", 0)
    )
    if n_kepler_total > 0 and not os.environ.get("KEPLER_RAW_DIR"):
        log.warning(
            "[build] requesting %d Kepler targets but KEPLER_RAW_DIR is unset. "
            "Files will land at %s (typically the internal SSD). "
            "Ctrl-C now and `export KEPLER_RAW_DIR=...` if that's not what you want.",
            n_kepler_total,
            paths.data_raw_kepler,
        )

    # --- Stage 1 — labelled catalogue ----------------------------------
    catalog = build_label_catalog(
        CatalogRequest(
            n_confirmed=int(cfg.data.n_confirmed),
            n_false_pos=int(cfg.data.n_false_pos),
            n_confirmed_kepler=int(cfg.data.get("n_confirmed_kepler", 0)),
            n_false_pos_kepler=int(cfg.data.get("n_false_pos_kepler", 0)),
            seed=int(cfg.data.seed),
        ),
        out_dir=paths.data_labels,
    )

    # Ensure every row has a mission column (backward compat with old catalogs).
    if "mission" not in catalog.columns:
        catalog["mission"] = "TESS"

    # Catalogue transit SNR kept as labels metadata (the aux vector now uses
    # the light-curve pink-noise SNR instead — see the aux comment below).
    catalog = enrich_catalog_snr(catalog, paths.root / "data" / "catalogue" / "candidates.parquet")
    catalog.to_parquet(paths.data_labels / "labels.parquet", index=False)

    # --- Stage 2 — download light curves -------------------------------
    kepler_dir = paths.data_raw_kepler if paths.data_raw_kepler != paths.data_raw else None
    downloader = LightCurveDownloader(
        cache_dir=paths.data_raw,
        kepler_cache_dir=kepler_dir,
        author=str(cfg.data.author),
        cadence=int(cfg.data.cadence) if cfg.data.cadence else None,
    )
    results = downloader.download_many(
        catalog["tic_id"].tolist(),
        missions=catalog["mission"].tolist(),
    )
    success_ids = {(r.mission, r.target_id) for r in results if r.success}
    log.info("[build] %d/%d targets downloaded successfully", len(success_ids), len(catalog))

    # --- Stage 3 — preprocess into views -------------------------------
    import lightkurve as lk

    g_views: list[np.ndarray] = []
    l_views: list[np.ndarray] = []
    labels: list[int] = []
    tic_ids: list[int] = []
    aux: list[list[float]] = []

    # Track *why* targets get dropped — invaluable on a long overnight build.
    # Catalog units are now days throughout (see catalog._query_toi docstring).
    skips = {
        "no_download": 0,
        "missing_ephemeris": 0,
        "missing_fits": 0,
        "preprocess_error": 0,
    }

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="processing"):
        tic = int(row["tic_id"])
        mission = row.get("mission", "TESS")
        if (mission, tic) not in success_ids:
            skips["no_download"] += 1
            continue
        period = row.get("period")
        t0 = row.get("t0")
        duration = row.get("duration")
        if (
            period is None
            or np.isnan(period)
            or t0 is None
            or np.isnan(t0)
            or duration is None
            or np.isnan(duration)
        ):
            skips["missing_ephemeris"] += 1
            continue

        # Resolve the FITS path based on mission.
        if mission == "Kepler":
            path = (kepler_dir or paths.data_raw) / f"kic_{tic}.fits"
        else:
            path = paths.data_raw / f"tic_{tic}.fits"
        if not path.exists():
            skips["missing_fits"] += 1
            continue
        try:
            lc = lk.read(str(path))
            # Centroid offset must come from the RAW light curve — clean/
            # flatten drop the MOM_CENTR1/2 columns it needs. A failure flows
            # to the imputer as NaN rather than killing the row.
            try:
                centroid_snr = float(
                    extract_centroid_offset(lc, float(period), float(t0), float(duration))
                )
            except Exception as cexc:
                log.debug("[build] %s %d centroid extract failed: %s", mission, tic, cexc)
                centroid_snr = float("nan")
            lc = clean_lightcurve(lc, sigma_clip=float(cfg.preprocess.cleaning.sigma_clip))
            # Ephemeris is known from the catalog row; mask transits when
            # fitting the flattening spline so the dip itself survives.
            lc = flatten_lightcurve(
                lc,
                window_length=int(cfg.preprocess.flatten.window_length),
                polyorder=int(cfg.preprocess.flatten.polyorder),
                period=float(period),
                t0=float(t0),
                duration=float(duration),
            )
            views = build_views(
                lc,
                period=float(period),
                t0=float(t0),
                duration=float(duration),
                global_bins=int(cfg.preprocess.views.global_bins),
                local_bins=int(cfg.preprocess.views.local_bins),
                local_durations=float(cfg.preprocess.views.local_durations),
            )
        except Exception as exc:
            log.warning("[build] TIC %d: preprocessing failed — %s", tic, exc)
            skips["preprocess_error"] += 1
            continue

        # Aux feature vector (13 dims — centroid stays at aux_transform.
        # CENTROID_COL == 8; the serving path branches on aux_dim so models
        # trained on the legacy 9-dim layout keep working):
        #   [teff, radius, logg, tmag, depth, duration, log_period,
        #    pink_snr, centroid_snr, oe_depth_sigma, oe_timing_sigma,
        #    secondary_sig, q_ratio]
        # Stellar params come from the KOI catalog row for Kepler targets and
        # from a TIC lookup for TESS targets. pink_snr (Kunimoto 2025 §2.1)
        # replaces the catalogue transit SNR: computed from the light curve,
        # it exists for every target at train AND serve time, closing the
        # non-TOI NaN->imputed-median mismatch. The vetting features come
        # from the same diagnostics the API serves as cautions.
        log_period = np.log(float(period)) if float(period) > 0 else np.nan
        depth_val = float(row["depth"]) if pd.notna(row.get("depth")) else np.nan
        dur_val = float(row["duration"]) if pd.notna(row.get("duration")) else np.nan
        if mission == "Kepler":
            stellar = [
                float(row["teff"]) if pd.notna(row.get("teff")) else np.nan,
                float(row["radius"]) if pd.notna(row.get("radius")) else np.nan,
                float(row["logg"]) if pd.notna(row.get("logg")) else np.nan,
                float(row["tmag"]) if pd.notna(row.get("tmag")) else np.nan,
            ]
        else:
            sp = fetch_stellar_params(tic)
            stellar = [
                sp.teff if sp.teff is not None else np.nan,
                sp.radius if sp.radius is not None else np.nan,
                sp.logg if sp.logg is not None else np.nan,
                sp.tmag if sp.tmag is not None else np.nan,
            ]
        t_arr = np.asarray(lc.time.value, dtype=float)
        f_arr = np.asarray(lc.flux.value, dtype=float)
        pn = pink_noise_snr(t_arr, f_arr, float(period), float(t0), float(duration))
        oe = odd_even_depths(t_arr, f_arr, float(period), float(t0), float(duration))
        sec = significant_secondary(t_arr, f_arr, float(period), float(t0), float(duration))
        dc = unphysical_duration(
            float(period), float(duration), stellar_radius=stellar[1], stellar_logg=stellar[2]
        )
        aux.append(
            [
                *stellar,
                depth_val,
                dur_val,
                log_period,
                pn.snr if pn is not None else np.nan,
                centroid_snr,
                oe.depth_diff_sigma if oe is not None else np.nan,
                oe.timing_diff_sigma
                if oe is not None and oe.timing_diff_sigma is not None
                else np.nan,
                sec.secondary_significance if sec is not None else np.nan,
                dc.q_ratio if dc is not None and dc.q_ratio is not None else np.nan,
            ]
        )
        g_views.append(views.global_view)
        l_views.append(views.local_view)
        labels.append(int(row["label"]))
        tic_ids.append(tic)

    log.info(
        "[build] kept %d targets;  skipped: no_download=%d  missing_ephemeris=%d  "
        "missing_fits=%d  preprocess_error=%d",
        len(g_views),
        skips["no_download"],
        skips["missing_ephemeris"],
        skips["missing_fits"],
        skips["preprocess_error"],
    )

    if not g_views:
        log.error("[build] no usable targets — check downloads and label catalogue")
        sys.exit(1)

    out = paths.data_processed / "views.npz"
    np.savez_compressed(
        out,
        global_views=np.stack(g_views),
        local_views=np.stack(l_views),
        labels=np.asarray(labels, dtype=np.int8),
        tic_ids=np.asarray(tic_ids, dtype=np.int64),
        aux_features=np.asarray(aux, dtype=np.float32),
    )
    log.info(
        "[build] wrote %d examples → %s  (pos=%d  neg=%d)",
        len(labels),
        out,
        int(np.sum(np.asarray(labels) == 1)),
        int(np.sum(np.asarray(labels) == 0)),
    )


if __name__ == "__main__":
    main()
