"""Tests for the TRICERATOPS statistical-validation wrapper.

The heavy dependency + its network calls are never imported: the pure helpers
are tested directly and the orchestration is tested against a fake target class.
"""

import numpy as np
import pandas as pd
import pytest

from exoplanet_hunter.validation import statistical as sv


@pytest.mark.parametrize(
    "fpp,nfpp,expected",
    [
        (0.005, 1e-4, sv.VALIDATED_PLANET),
        (0.014, 9e-4, sv.VALIDATED_PLANET),
        (0.015, 1e-4, sv.LIKELY_PLANET),  # FPP boundary is strict
        (0.2, 1e-4, sv.LIKELY_PLANET),
        (0.9, 1e-4, sv.LIKELY_FP),  # target-side FP, low NFPP
        (0.001, 0.5, sv.LIKELY_NEARBY_FP),
        (0.9, 0.5, sv.LIKELY_NEARBY_FP),  # NFPP takes precedence over FPP
        (0.2, 0.05, sv.INCONCLUSIVE),  # mid NFPP, sub-0.5 FPP
    ],
)
def test_classify_covers_every_branch(fpp, nfpp, expected):
    assert sv.classify(fpp, nfpp) == expected


def test_estimate_snr():
    assert sv.estimate_snr(1000.0, 100.0, 9) == pytest.approx(30.0)  # 10 * 3
    assert sv.estimate_snr(100.0, 100.0, 4) == pytest.approx(2.0)  # unreliable (<15)
    assert sv.estimate_snr(1000.0, 0.0, 9) is None
    assert sv.estimate_snr(1000.0, 100.0, 0) is None


def _boxed_transit(period=3.0, t0=1.0, duration=0.1, depth=0.01, n=40, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0, period * n, period / 300)
    flux = 2.0 + rng.normal(0, 1e-4, time.size)  # baseline 2.0 -> tests normalisation
    dt = np.mod(time - t0 + 0.5 * period, period) - 0.5 * period
    flux[np.abs(dt) < duration / 2] -= depth * 2.0
    return time, flux


def test_prepare_lightcurve_folds_normalises_and_windows():
    time, flux = _boxed_transit()
    dt, norm, sigma = sv.prepare_lightcurve(time, flux, period=3.0, t0=1.0, duration=0.1)

    assert np.all(np.abs(dt) < 5 * 0.1 + 1e-9)  # within the window
    assert np.all(np.diff(dt) >= 0)  # sorted by phase
    assert np.median(norm[np.abs(dt) > 0.1]) == pytest.approx(1.0, abs=2e-3)  # baseline -> 1
    assert norm[np.argmin(np.abs(dt))] < 0.995  # transit dip survives
    assert 0 < sigma < 1e-3


def test_prepare_lightcurve_rejects_bad_inputs():
    time, flux = _boxed_transit()
    with pytest.raises(ValueError):
        sv.prepare_lightcurve(time, flux, period=0.0, t0=1.0, duration=0.1)


class _FakeTarget:
    """Stand-in for triceratops.triceratops.target (no network, no pixels)."""

    def __init__(self, ID, sectors, mission, search_radius, trilegal_fname=None):
        self.ID = ID
        self.sectors = sectors
        self.trilegal_fname = trilegal_fname
        self.calls: dict = {}
        self.stars = pd.DataFrame({"ID": [ID, 111, 222]})  # target + 2 neighbours

    def calc_depths(self, tdepth, all_ap_pixels=None):
        self.calls["tdepth"] = tdepth

    def calc_probs(self, time, flux_0, flux_err_0, P_orb, **kwargs):
        self.calls.update(P_orb=P_orb, n_points=len(time), kwargs=kwargs)
        self.probs = pd.DataFrame({"scenario": ["TP", "EB", "NEB"], "prob": [0.97, 0.02, 0.01]})
        self.FPP = 0.03
        self.NFPP = 0.01


def test_validate_target_orchestrates_and_classifies(monkeypatch):
    monkeypatch.setattr(sv, "_load_target_cls", lambda: _FakeTarget)
    dt, norm, sigma = sv.prepare_lightcurve(*_boxed_transit(), period=3.0, t0=1.0, duration=0.1)
    result = sv.validate_target(
        tic_id=12345,
        sectors=np.array([1, 2]),
        period_days=3.0,
        depth_ppm=10_000.0,
        phase_time=dt,
        flux=norm,
        flux_err=sigma,
        snr=25.0,
        n_draws=1000,
    )
    assert result.fpp == 0.03 and result.nfpp == 0.01
    assert result.classification == sv.INCONCLUSIVE  # NFPP 0.01 in the mid band
    assert result.best_scenario == "TP"
    assert result.n_nearby_stars == 3
    assert result.snr_reliable is True
    assert result.scenario_probs["NEB"] == 0.01


def test_compat_shims_restore_removed_names():
    # pytransit imports names that modern numpy/scipy/setuptools dropped; the
    # shim restores them so `import triceratops` doesn't die in a dependency.
    import scipy.integrate as si

    sv._install_triceratops_compat_shims()
    assert np.int is int  # noqa: NPY001 — NumPy 1.24 alias restored by the shim
    assert si.trapz is si.trapezoid  # SciPy trapz->trapezoid bridged
    import pkg_resources  # real (setuptools<81) or our stub

    assert hasattr(pkg_resources, "resource_filename")
    sv._install_triceratops_compat_shims()  # idempotent


def test_trilegal_ssl_disabled_forces_verify_off(monkeypatch):
    import sys
    import types

    calls = []

    def query_TRILEGAL(ra, dec, verbose=0, verify_ssl=True):
        calls.append(verify_ssl)
        return "url"

    fake_tt = types.ModuleType("triceratops.triceratops")
    fake_tt.query_TRILEGAL = query_TRILEGAL
    monkeypatch.setitem(sys.modules, "triceratops", types.ModuleType("triceratops"))
    monkeypatch.setitem(sys.modules, "triceratops.triceratops", fake_tt)
    original = fake_tt.query_TRILEGAL

    with sv._trilegal_ssl_disabled():
        fake_tt.query_TRILEGAL(10.0, 20.0, verify_ssl=True)  # target() passes True
    assert calls == [False]  # ...but the patch forced verification off
    assert fake_tt.query_TRILEGAL is original  # restored on exit


def test_validate_target_raises_helpful_error_without_dep(monkeypatch):
    def _boom():
        raise ImportError("pip install -e 'pipeline[validation]'")

    monkeypatch.setattr(sv, "_load_target_cls", _boom)
    with pytest.raises(ImportError, match="pipeline\\[validation\\]"):
        sv.validate_target(
            tic_id=1,
            sectors=np.array([1]),
            period_days=3.0,
            depth_ppm=1000.0,
            phase_time=np.linspace(-0.2, 0.2, 50),
            flux=np.ones(50),
            flux_err=1e-4,
        )
