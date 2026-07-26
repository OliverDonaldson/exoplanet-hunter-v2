"""Statistical validation of shortlist candidates with TRICERATOPS.

Our CNN ranks candidates by a calibrated probability but never reads the pixels,
so it cannot separate a transit on the target from an eclipse on a nearby star
bleeding into the aperture. TRICERATOPS (Giacalone et al. 2021, AJ 161:24)
closes that gap with a Bayesian model over 15 target-star scenarios (planet / EB
/ period-doubled EB on the target, on an unresolved bound companion, or on an
unresolved background star) plus nearby-star scenarios (NTP/NEB/NEBx2P for each
resolved neighbour), scored against the phase-folded light curve and the TESS
pixel data. It yields two numbers:

  * ``FPP``  = 1 - (P_TP + P_PTP + P_DTP): probability the signal is NOT a planet
    transiting the target star (Eq 4);
  * ``NFPP`` = sum of the nearby-star scenario probabilities: probability the
    signal originates from a resolved neighbour (Eq 5) — the piece a
    light-curve-only vetter (ours included) is blind to.

This is a slow, network-bound, OFFLINE step (TIC cone search, pixel cutout,
TRILEGAL galactic model, ~1e6 Monte-Carlo draws): a validation pass over a
ranked shortlist, never part of live ``/score``. ``triceratops`` is an optional
dependency — ``pip install -e 'pipeline[validation]'``.

Caveats carried straight from the paper:
  * feed SIMPLE-APERTURE (SAP) flux, not PDCSAP — PDC removes the very
    nearby-star contamination NFPP exists to catch;
  * FPP is unreliable below S/N ~15 (Eq 17); :func:`estimate_snr` flags it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: Validated-planet thresholds (Giacalone 2021, §4): NFPP < 1e-3 and FPP < 0.015.
FPP_VALIDATED = 0.015
#: A candidate with FPP at or above this is more likely a false positive.
FPP_LIKELY = 0.5
NFPP_VALIDATED = 1e-3
#: Above this NFPP the transit most likely comes from a resolved neighbour.
NFPP_NEARBY_FP = 0.1
#: FPP predictions are unreliable below this transit S/N (Eq 17).
SNR_RELIABLE_MIN = 15.0

VALIDATED_PLANET = "validated_planet"
LIKELY_PLANET = "likely_planet"
LIKELY_NEARBY_FP = "likely_nearby_fp"
LIKELY_FP = "likely_fp"
INCONCLUSIVE = "inconclusive"


def classify(fpp: float, nfpp: float) -> str:
    """Map an (FPP, NFPP) pair to a disposition (Giacalone 2021, §4).

    A high NFPP means a resolved neighbour is the likely source, so it takes
    precedence; otherwise a high FPP means a target-side false positive (e.g. an
    EB on the target). Only a low NFPP *and* low FPP validates a planet.
    """
    if nfpp > NFPP_NEARBY_FP:
        return LIKELY_NEARBY_FP
    if fpp >= FPP_LIKELY:
        return LIKELY_FP
    if nfpp < NFPP_VALIDATED and fpp < FPP_VALIDATED:
        return VALIDATED_PLANET
    if nfpp < NFPP_VALIDATED and fpp < FPP_LIKELY:
        return LIKELY_PLANET
    return INCONCLUSIVE  # 1e-3 <= NFPP <= 0.1 with FPP < 0.5 — needs follow-up


def estimate_snr(depth_ppm: float, cdpp_ppm: float, n_transits: int) -> float | None:
    """Transit S/N (Giacalone 2021, Eq 17): delta_obs / sigma_CDPP · sqrt(n_tra).

    None when the inputs cannot form a ratio. TRICERATOPS FPP values below
    S/N ~15 are unreliable — false positives frequently score a low FPP there.
    """
    if not (cdpp_ppm > 0 and n_transits > 0 and np.isfinite(depth_ppm)):
        return None
    return float(depth_ppm / cdpp_ppm * np.sqrt(n_transits))


def prepare_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    *,
    window_durations: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fold a light curve into the (time-from-midpoint, normalised flux, sigma)
    triple TRICERATOPS' ``calc_probs`` expects.

    Folds to time from the nearest transit midpoint (days), keeps points within
    ``window_durations`` transit durations of it, normalises flux by the
    out-of-transit median (baseline 1.0), and returns a scalar flux uncertainty
    from the robust out-of-transit scatter (1.4826·MAD). Points are sorted by
    phase. Feed SAP flux, not PDCSAP (see the module docstring).
    """
    ok = np.isfinite(time) & np.isfinite(flux)
    t, f = np.asarray(time, float)[ok], np.asarray(flux, float)[ok]
    if period <= 0 or duration <= 0 or t.size == 0:
        raise ValueError("prepare_lightcurve needs positive period/duration and data")

    # Time from the nearest transit midpoint, in [-P/2, P/2).
    dt = np.mod(t - t0 + 0.5 * period, period) - 0.5 * period
    keep = np.abs(dt) < window_durations * duration
    if keep.sum() < 3:
        raise ValueError("no in-window points after folding")
    dt, f = dt[keep], f[keep]
    order = np.argsort(dt)
    dt, f = dt[order], f[order]

    oot = np.abs(dt) > duration
    baseline = float(np.median(f[oot])) if oot.sum() >= 3 else float(np.median(f))
    if baseline == 0:
        raise ValueError("zero baseline flux — cannot normalise")
    norm = f / baseline
    resid = norm[oot] - 1.0 if oot.sum() >= 3 else norm - float(np.median(norm))
    sigma = float(1.4826 * np.median(np.abs(resid))) or float(np.std(norm))
    return dt, norm, sigma


@dataclass(frozen=True)
class StatisticalValidation:
    """TRICERATOPS result for one candidate."""

    tic_id: int
    fpp: float
    nfpp: float
    classification: str
    best_scenario: str
    scenario_probs: dict[str, float]
    n_nearby_stars: int
    snr: float | None = None
    snr_reliable: bool | None = None
    contrast_curve_used: bool = False


def _install_triceratops_compat_shims() -> None:
    """Make pytransit (a TRICERATOPS dependency) importable under this env's
    modern numpy / scipy / setuptools. pytransit 2.2.0 predates three removals:

      * ``numpy.int`` — dropped in NumPy 1.24 (restore the alias to the builtin);
      * ``scipy.integrate.trapz`` — renamed ``trapezoid``, dropped in SciPy 1.14;
      * ``pkg_resources`` — dropped in setuptools 81; pytransit's contamination
        module imports only ``resource_filename`` from it, so stub that.

    Each shim is a no-op when the real name is present, so a compatible env is
    left untouched. Without these, ``import triceratops`` dies deep in pytransit
    with an unhelpful ImportError.
    """
    import numpy as _np

    if not hasattr(_np, "int"):
        _np.int = int  # type: ignore[attr-defined]  # noqa: NPY001 — intentional legacy-alias restore

    import scipy.integrate as _si

    if not hasattr(_si, "trapz"):
        _si.trapz = _si.trapezoid

    import importlib.util
    import sys

    # find_spec raises on our spec-less stub, so short-circuit once it's installed.
    if "pkg_resources" not in sys.modules and importlib.util.find_spec("pkg_resources") is None:
        import os
        import types

        def _resource_filename(package: str, resource: str) -> str:
            base = os.path.dirname(importlib.import_module(package).__file__ or "")
            return os.path.join(base, resource)

        stub = types.ModuleType("pkg_resources")
        stub.resource_filename = _resource_filename  # type: ignore[attr-defined]
        stub.get_distribution = lambda name: types.SimpleNamespace(version="0")  # type: ignore[attr-defined]
        stub.DistributionNotFound = Exception  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = stub


def _load_target_cls() -> type:
    _install_triceratops_compat_shims()
    try:
        from triceratops.triceratops import target
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing == "triceratops":
            raise ImportError(
                "TRICERATOPS is not installed. Install the validation extra:\n"
                "    pip install -e 'pipeline[validation]'"
            ) from exc
        raise ImportError(
            f"TRICERATOPS is installed but its dependency {missing!r} failed to "
            "import in this environment."
        ) from exc
    return target


@contextlib.contextmanager
def _trilegal_ssl_disabled() -> Iterator[None]:
    """Scope an SSL-verification bypass to TRICERATOPS' TRILEGAL query only.

    The TRILEGAL server (stev.oapd.inaf.it) ships an incomplete certificate
    chain that even an up-to-date CA bundle cannot verify — a known issue, which
    is why TRICERATOPS gave ``query_TRILEGAL`` a ``verify_ssl`` flag. But
    ``target()`` hardcodes ``verify_ssl=True`` and never exposes it, so we patch
    the reference it calls to force verification off, then restore it. The query
    is a public, unauthenticated star-count lookup (only RA/Dec is sent), so
    skipping verification here carries no data-exposure risk.
    """
    import triceratops.triceratops as _tt

    original = _tt.query_TRILEGAL

    def _no_verify(ra: float, dec: float, *args: object, **kwargs: object) -> object:
        kwargs["verify_ssl"] = False
        return original(ra, dec, *args, **kwargs)

    _tt.query_TRILEGAL = _no_verify
    try:
        yield
    finally:
        _tt.query_TRILEGAL = original


def _aperture_to_cutout_pixels(
    pipeline_mask: np.ndarray,
    target_xy_tpf: tuple[float, float],
    target_xy_cutout: np.ndarray,
) -> np.ndarray:
    """Map a SPOC pipeline aperture into TRICERATOPS' TessCut cutout frame.

    ``pipeline_mask`` is the boolean [row, col] SPOC aperture on the TPF grid;
    ``target_xy_tpf`` the target's [col, row] position on that grid;
    ``target_xy_cutout`` its [col, row] position in TRICERATOPS' cutout. Both
    products sample the same native CCD pixels, so each aperture pixel's offset
    from the target is frame-invariant — shift those offsets onto the target's
    cutout position. Returns the (N, 2) [col, row] array ``calc_depths`` expects.
    """
    rows, cols = np.nonzero(pipeline_mask)
    dcol = cols - target_xy_tpf[0]
    drow = rows - target_xy_tpf[1]
    return np.column_stack([target_xy_cutout[0] + dcol, target_xy_cutout[1] + drow])


def _fetch_pipeline_apertures(tgt: object, tic_id: int, sectors: np.ndarray) -> list | None:
    """The real SPOC photometric aperture per sector, in TRICERATOPS' cutout frame.

    TRICERATOPS otherwise assumes a 5x5 box, which admits more contaminant flux
    than the pipeline aperture and inflates FPP. Keyed to ``tgt.pix_coords`` so
    sector order matches; returns None on any gap (missing TPF, mask, or a
    sector-count mismatch) so the caller cleanly falls back to the 5x5 default.
    """
    import lightkurve as lk

    pix_coords = getattr(tgt, "pix_coords", None)
    stars = getattr(tgt, "stars", None)
    if pix_coords is None or stars is None or len(pix_coords) != len(sectors):
        return None
    ra, dec = float(stars.iloc[0]["ra"]), float(stars.iloc[0]["dec"])
    apertures = []
    for k, sector in enumerate(sectors):
        search = lk.search_targetpixelfile(
            f"TIC {tic_id}", mission="TESS", author="SPOC", sector=int(sector)
        )
        if len(search) == 0:
            return None
        tpf = search.download()
        mask = np.asarray(tpf.pipeline_mask)
        if not mask.any():
            return None
        xt, yt = tpf.wcs.all_world2pix(ra, dec, 0)
        apertures.append(_aperture_to_cutout_pixels(mask, (float(xt), float(yt)), pix_coords[k][0]))
    return apertures


def validate_target(
    *,
    tic_id: int,
    sectors: np.ndarray,
    period_days: float,
    depth_ppm: float,
    phase_time: np.ndarray,
    flux: np.ndarray,
    flux_err: float,
    contrast_curve_file: str | None = None,
    mission: str = "TESS",
    search_radius: int = 10,
    n_draws: int = 1_000_000,
    parallel: bool = False,
    apertures: list[np.ndarray] | None = None,
    use_pipeline_aperture: bool = True,
    snr: float | None = None,
    trilegal_fname: str | None = None,
    verify_ssl: bool = True,
    verbose: int = 0,
) -> StatisticalValidation:
    """Run TRICERATOPS for one target and return its FPP/NFPP disposition.

    ``phase_time`` is days from the transit midpoint, ``flux`` normalised to a
    baseline of 1, ``flux_err`` a scalar (all as produced by
    :func:`prepare_lightcurve`); ``depth_ppm`` seeds the per-star required-depth
    calculation. Constructing the target hits the network (TIC + pixel cutout +
    a TRILEGAL galactic-model query), so this is never called on live serving.

    ``use_pipeline_aperture`` (TESS) fetches the real SPOC aperture instead of
    TRICERATOPS' 5x5 default, which otherwise inflates FPP; it falls back to the
    default (with a warning) if the aperture can't be built. ``apertures``
    overrides it. Two escapes for TRILEGAL's broken TLS cert: pass
    ``trilegal_fname`` (pre-downloaded table, fully secure) or opt into
    ``verify_ssl=False`` to bypass verification for that one public query.
    """
    target_cls = _load_target_cls()
    ssl_ctx = (
        contextlib.nullcontext()
        if verify_ssl or trilegal_fname is not None
        else _trilegal_ssl_disabled()
    )
    with ssl_ctx:
        tgt = target_cls(
            ID=int(tic_id),
            sectors=np.asarray(sectors, dtype=int),
            mission=mission,
            search_radius=search_radius,
            trilegal_fname=trilegal_fname,
        )
    if apertures is None and use_pipeline_aperture and mission == "TESS":
        try:
            apertures = _fetch_pipeline_apertures(tgt, int(tic_id), np.asarray(sectors))
        except Exception as exc:
            log.warning("[validation] SPOC aperture fetch failed for TIC %d: %s", tic_id, exc)
            apertures = None
        if apertures is None:
            log.warning(
                "[validation] TIC %d: no SPOC pipeline aperture — TRICERATOPS 5x5 "
                "default (FPP will read looser/higher)",
                tic_id,
            )
    # calc_depths' docstring says ppm, but its arithmetic needs a fraction:
    # it computes tdepth/fluxratio per star and then zeroes anything > 1. Fed
    # ppm, every star zeroes, leaving 12 scenarios with no evidence computed —
    # a uniform 1/12 each, so FPP is exactly 1-3/12 = 0.75 and NFPP falls to
    # the hardcoded 0.0 branch, for every target.
    tgt.calc_depths(tdepth=float(depth_ppm) / 1e6, all_ap_pixels=apertures)
    tgt.calc_probs(
        time=np.asarray(phase_time, dtype=float),
        flux_0=np.asarray(flux, dtype=float),
        flux_err_0=float(flux_err),
        P_orb=float(period_days),
        contrast_curve_file=contrast_curve_file,
        N=int(n_draws),
        parallel=parallel,
        verbose=verbose,
    )
    probs = tgt.probs
    fpp, nfpp = float(tgt.FPP), float(tgt.NFPP)
    scenario_probs = {
        str(s): float(p) for s, p in zip(probs["scenario"], probs["prob"], strict=False)
    }
    # Highest-probability *named* scenario — TRICERATOPS leaves the scenario blank
    # on padding rows, so filter those out rather than report an empty label.
    named = {s: p for s, p in scenario_probs.items() if s.strip()}
    best_scenario = max(named, key=lambda s: named[s], default="")
    return StatisticalValidation(
        tic_id=int(tic_id),
        fpp=fpp,
        nfpp=nfpp,
        classification=classify(fpp, nfpp),
        best_scenario=best_scenario,
        scenario_probs=scenario_probs,
        n_nearby_stars=len(tgt.stars),
        snr=snr,
        snr_reliable=(snr >= SNR_RELIABLE_MIN) if snr is not None else None,
        contrast_curve_used=contrast_curve_file is not None,
    )
