"""Regenerate the data-provenance sky figures from the live artefacts.

Sky positions come from the RA_OBJ/DEC_OBJ keywords of each target's cached
FITS file, so the figures only ever show targets actually downloaded.

    python pipeline/scripts/plot_provenance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from exoplanet_hunter.utils import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "figures"

# mission -> (cache dir, filename prefix, colour, marker size)
MISSIONS = {
    "TESS": ("data/raw/tess/lightcurves", "tic", "royalblue", 2.0),
    "Kepler": ("data/raw/kepler/lightcurves", "kic", "crimson", 2.0),
    "K2": ("data/raw/tess/lightcurves", "epic", "darkorange", 3.0),
}


def _resolve_positions(catalog: pd.DataFrame) -> pd.DataFrame:
    """Attach RA/Dec from the cached FITS headers; drop unresolved targets."""
    ra, dec = [], []
    for tid, mission in zip(catalog["tic_id"], catalog["mission"], strict=True):
        sub, prefix, *_ = MISSIONS[mission]
        path = REPO_ROOT / sub / f"{prefix}_{int(tid)}.fits"
        try:
            header = fits.getheader(path, 0)
            ra.append(float(header["RA_OBJ"]))
            dec.append(float(header["DEC_OBJ"]))
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            ra.append(np.nan)
            dec.append(np.nan)
    out = catalog.assign(ra_deg=ra, dec_deg=dec).dropna(subset=["ra_deg", "dec_deg"])
    log.info("[provenance] resolved %d/%d target positions", len(out), len(catalog))
    return out


def _mollweide(ax) -> None:
    ax.grid(color="0.85", lw=0.5)
    ax.set_xlabel("Right Ascension →", fontsize=9)
    ax.tick_params(labelsize=8)


def _radians(ra_deg, dec_deg):
    # Mollweide wants longitude in [-pi, pi]; wrap RA about 0h so the map is
    # centred on RA=0 (Kepler's Cygnus field then sits left of centre).
    ra = np.asarray(ra_deg, dtype=float)
    return np.radians(np.where(ra > 180.0, ra - 360.0, ra)), np.radians(np.asarray(dec_deg))


def plot_sky_map(pos: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(13, 7.6))
    ax = fig.add_subplot(111, projection="mollweide")
    _mollweide(ax)
    handles = []
    for mission, (_, _, colour, size) in MISSIONS.items():
        sub = pos[pos["mission"] == mission]
        if sub.empty:
            continue
        x, y = _radians(sub["ra_deg"], sub["dec_deg"])
        ax.scatter(x, y, s=size, c=colour, alpha=0.75, linewidths=0, zorder=3)
        handles.append((mission, len(sub), colour, size))

    ax.set_title(
        "Where the model has looked — training targets on the sky (equatorial, Mollweide)",
        fontsize=12,
        pad=18,
    )
    footprints = {
        "TESS": "all-sky",
        "Kepler": "one field in Cygnus",
        "K2": "ecliptic plane",
    }
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=c,
                markersize=max(3.5, s),
                label=f"{m}  (n={n:,}, {footprints[m]})",
            )
            for m, n, c, s in handles
        ],
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("[provenance] wrote %s", out)


def plot_coverage_map(pos: pd.DataFrame, candidates: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(13, 7.6))
    ax = fig.add_subplot(111, projection="mollweide")
    _mollweide(ax)

    pool = candidates.dropna(subset=["ra_deg", "dec_deg"])
    xp, yp = _radians(pool["ra_deg"], pool["dec_deg"])
    ax.scatter(xp, yp, s=1.0, c="0.80", alpha=0.55, linewidths=0, zorder=1)

    handles = [("TESS candidate pool available", len(pool), "0.80", 1.0)]
    for mission, (_, _, colour, size) in MISSIONS.items():
        sub = pos[pos["mission"] == mission]
        if sub.empty:
            continue
        x, y = _radians(sub["ra_deg"], sub["dec_deg"])
        ax.scatter(x, y, s=size, c=colour, alpha=0.8, linewidths=0, zorder=3)
        handles.append((f"trained on — {mission}", len(sub), colour, size))

    ax.set_title(
        "What the model trained on  vs  what has been flagged as candidates",
        fontsize=12,
        pad=18,
    )
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=c,
                markersize=max(3.5, s),
                label=f"{lbl}  (n={n:,})",
            )
            for lbl, n, c, s in handles
        ],
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("[provenance] wrote %s", out)


def main() -> None:
    catalog = pd.read_parquet(REPO_ROOT / "data" / "labels" / "labels.parquet")
    candidates = pd.read_parquet(REPO_ROOT / "data" / "catalogue" / "candidates.parquet")
    pos = _resolve_positions(catalog)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_sky_map(pos, FIG_DIR / "sky_map.png")
    plot_coverage_map(pos, candidates, FIG_DIR / "coverage_map.png")

    counts = pos["mission"].value_counts().to_dict()
    summary = {
        "labelled_targets": len(catalog),
        "resolved_positions": len(pos),
        "by_mission_labelled": catalog["mission"].value_counts().to_dict(),
        "by_mission_resolved": counts,
        "candidate_pool": len(candidates),
    }
    log.info("[provenance] %s", json.dumps(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
