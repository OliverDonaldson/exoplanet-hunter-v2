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

from dataclasses import dataclass

import numpy as np

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

    if importlib.util.find_spec("pkg_resources") is None:
        import os
        import sys
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
    snr: float | None = None,
    verbose: int = 0,
) -> StatisticalValidation:
    """Run TRICERATOPS for one target and return its FPP/NFPP disposition.

    ``phase_time`` is days from the transit midpoint, ``flux`` normalised to a
    baseline of 1, ``flux_err`` a scalar (all as produced by
    :func:`prepare_lightcurve`); ``depth_ppm`` seeds the per-star required-depth
    calculation. Constructing the target hits the network (TIC + pixel cutout),
    so this is never called on the live serving path.
    """
    target_cls = _load_target_cls()
    tgt = target_cls(
        ID=int(tic_id),
        sectors=np.asarray(sectors, dtype=int),
        mission=mission,
        search_radius=search_radius,
    )
    tgt.calc_depths(tdepth=float(depth_ppm), all_ap_pixels=apertures)
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
    best = probs.loc[probs["prob"].idxmax()]
    return StatisticalValidation(
        tic_id=int(tic_id),
        fpp=fpp,
        nfpp=nfpp,
        classification=classify(fpp, nfpp),
        best_scenario=str(best["scenario"]),
        scenario_probs={
            str(s): float(p) for s, p in zip(probs["scenario"], probs["prob"], strict=False)
        },
        n_nearby_stars=len(tgt.stars),
        snr=snr,
        snr_reliable=(snr >= SNR_RELIABLE_MIN) if snr is not None else None,
        contrast_curve_used=contrast_curve_file is not None,
    )
