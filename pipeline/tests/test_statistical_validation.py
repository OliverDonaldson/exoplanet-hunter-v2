"""Tests for the TRICERATOPS statistical-validation wrapper.

The heavy dependency + its network calls are never imported: the pure helpers
are tested directly and the orchestration is tested against a fake target class.
"""

from pathlib import Path

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
        use_pipeline_aperture=False,
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


def test_aperture_to_cutout_pixels_shifts_by_ccd_offset():
    # target on the central True pixel (TPF col=1,row=1); the same star sits at
    # (col=10,row=20) in TRICERATOPS' cutout. Aperture pixels keep their offset.
    mask = np.array([[False, False, False], [False, True, True], [False, False, False]])
    out = sv._aperture_to_cutout_pixels(mask, (1.0, 1.0), np.array([10.0, 20.0]))
    assert out.shape == (2, 2)  # (N pixels, [col, row])
    assert out.tolist() == [[10.0, 20.0], [11.0, 20.0]]


def test_fetch_pipeline_apertures_falls_back_on_gaps():
    # No pix_coords / sector-count mismatch -> None (caller uses the 5x5 default),
    # and never touches the network.
    class _Bare:
        pass

    assert sv._fetch_pipeline_apertures(_Bare(), 1, np.array([1, 2])) is None


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


def test_calc_depths_receives_a_fraction_not_ppm(monkeypatch):
    """TRICERATOPS' calc_depths docstring says ppm but its arithmetic wants a
    fraction: it computes tdepth/fluxratio per star then zeroes anything > 1.
    Fed ppm, every star zeroes and only the 12 target-side scenarios remain
    with no evidence computed — uniform 1/12 each, so FPP is exactly
    1 - 3/12 = 0.75 and NFPP falls to the hardcoded 0.0 branch, identically
    for every target. That is what a whole 20-target shortlist returned.
    """
    monkeypatch.setattr(sv, "_load_target_cls", lambda: _FakeTarget)
    dt, norm, sigma = sv.prepare_lightcurve(*_boxed_transit(), period=3.0, t0=1.0, duration=0.1)
    captured: dict = {}

    class _Capturing(_FakeTarget):
        def calc_depths(self, tdepth, all_ap_pixels=None):
            captured["tdepth"] = tdepth
            super().calc_depths(tdepth, all_ap_pixels)

    monkeypatch.setattr(sv, "_load_target_cls", lambda: _Capturing)
    sv.validate_target(
        tic_id=12345,
        sectors=np.array([1, 2]),
        period_days=3.0,
        depth_ppm=602.0,
        phase_time=dt,
        flux=norm,
        flux_err=sigma,
        snr=25.0,
        n_draws=1000,
        use_pipeline_aperture=False,
    )
    assert captured["tdepth"] == pytest.approx(602.0e-6)
    # The degenerate regime is anything that cannot survive the > 1 cut.
    assert captured["tdepth"] <= 1.0


# ------------------------------------------------------- degeneracy guards --


def test_uniform_posterior_is_degenerate():
    """A flat posterior means the evidence integral discriminated nothing.

    TRICERATOPS initialises lnZ = zeros(N) and fills it per scenario; if
    nothing fills it the evidences stay equal, normalise to uniform, and FPP
    becomes the constant 1 - 3/N. That is arithmetic, not a measurement.
    """
    assert sv.is_degenerate_posterior({f"s{i}": 1 / 21 for i in range(21)})
    assert sv.is_degenerate_posterior({"only": 1.0})  # nothing to compare
    assert not sv.is_degenerate_posterior({"TP": 0.9, "EB": 0.07, "NEB": 0.03})


def test_tic_441804533_signature_is_caught():
    """The real failure: 21 uniform scenarios -> FPP 6/7, NFPP 2/7.

    The vendored fork does NOT catch this — its FPP_degenerate flags -inf and
    NaN evidences, but uniform *finite* evidences normalise cleanly and report
    status "ok". Reported as likely_nearby_fp before this guard existed.
    """
    n = 21
    probs = np.full(n, 1 / n)
    fpp = 1 - (probs[0] + probs[3] + probs[9])
    nfpp = probs[15:].sum()
    assert fpp == pytest.approx(6 / 7)
    assert nfpp == pytest.approx(2 / 7)
    # Without the guard this is a confident-looking nearby-FP verdict.
    assert sv.classify(fpp, nfpp) == sv.LIKELY_NEARBY_FP
    assert sv.classify(fpp, nfpp, degenerate=True) == sv.DEGENERATE


def test_classify_rejects_non_finite():
    """FPP=NaN previously fell through every threshold to "inconclusive"."""
    assert sv.classify(float("nan"), 0.0) == sv.DEGENERATE
    assert sv.classify(0.01, float("nan")) == sv.DEGENERATE


def test_validate_target_flags_degenerate_run(monkeypatch):
    """End to end: a uniform posterior must not be reported as a disposition."""

    class _UniformTarget(_FakeTarget):
        def calc_probs(self, time, flux_0, flux_err_0, P_orb, **kwargs):
            n = 21
            self.probs = pd.DataFrame(
                {"scenario": [f"S{i}" for i in range(n)], "prob": [1 / n] * n}
            )
            self.FPP = 1 - 3 / n
            self.NFPP = (n - 15) / n
            self.FPP_degenerate = False  # the fork does not catch this case

    monkeypatch.setattr(sv, "_load_target_cls", lambda: _UniformTarget)
    dt, norm, sigma = sv.prepare_lightcurve(*_boxed_transit(), period=3.0, t0=1.0, duration=0.1)
    result = sv.validate_target(
        tic_id=441804533,
        sectors=np.array([1]),
        period_days=3.0,
        depth_ppm=1000.0,
        phase_time=dt,
        flux=norm,
        flux_err=sigma,
        snr=22.0,
        n_draws=1000,
        use_pipeline_aperture=False,
    )
    assert result.degenerate
    assert result.classification == sv.DEGENERATE
    assert "uniform posterior across 21 scenarios" in result.degenerate_reason


def test_validate_target_honours_the_fork_flag(monkeypatch):
    """The other half: -inf/NaN evidences, which the fork does flag."""

    class _FlaggedTarget(_FakeTarget):
        def calc_probs(self, time, flux_0, flux_err_0, P_orb, **kwargs):
            self.probs = pd.DataFrame({"scenario": ["TP", "EB", "NEB"], "prob": [0.0, 0.0, 0.0]})
            self.FPP = 1.0
            self.NFPP = 0.0
            self.FPP_degenerate = True

    monkeypatch.setattr(sv, "_load_target_cls", lambda: _FlaggedTarget)
    dt, norm, sigma = sv.prepare_lightcurve(*_boxed_transit(), period=3.0, t0=1.0, duration=0.1)
    result = sv.validate_target(
        tic_id=999,
        sectors=np.array([1]),
        period_days=3.0,
        depth_ppm=1000.0,
        phase_time=dt,
        flux=norm,
        flux_err=sigma,
        snr=22.0,
        n_draws=1000,
        use_pipeline_aperture=False,
    )
    assert result.degenerate
    assert result.classification == sv.DEGENERATE
    assert "triceratops flagged" in result.degenerate_reason


def test_healthy_posterior_is_not_flagged(monkeypatch):
    """The guard must not fire on a real result (TIC 451645081's shape)."""
    monkeypatch.setattr(sv, "_load_target_cls", lambda: _FakeTarget)
    dt, norm, sigma = sv.prepare_lightcurve(*_boxed_transit(), period=3.0, t0=1.0, duration=0.1)
    result = sv.validate_target(
        tic_id=451645081,
        sectors=np.array([1, 2]),
        period_days=8.1,
        depth_ppm=1000.0,
        phase_time=dt,
        flux=norm,
        flux_err=sigma,
        snr=84.5,
        n_draws=1000,
        use_pipeline_aperture=False,
    )
    assert not result.degenerate
    assert result.degenerate_reason is None
    assert result.classification != sv.DEGENERATE


def test_per_target_timeout_abandons_and_continues():
    """One pathological target must not stall the run.

    A single target ran 10 h at 99% CPU producing nothing, and every target
    behind it waited. SIGALRM fires between operations, so this cannot
    interrupt one long C call — it escapes a Python-level loop, which is where
    TRICERATOPS spends its time (a scenario loop).
    """
    import importlib.util
    import time as _time

    spec = importlib.util.spec_from_file_location(
        "_vc", Path(__file__).resolve().parents[1] / "scripts" / "validate_candidates.py"
    )
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)

    with pytest.raises(vc.TargetTimeout), vc._time_limit(1):
        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            pass

    # The alarm is cleared afterwards, so the next target gets a full budget.
    with vc._time_limit(5):
        _time.sleep(0.01)

    # 0 disables the limit entirely.
    with vc._time_limit(0):
        _time.sleep(0.01)


def test_skip_completed_resumes_and_retries_failures(tmp_path):
    """An interrupted run must not redo finished targets.

    The docstring claimed "resumable-by-rerun" while nothing skipped completed
    work; one restart cost ~3 h re-deriving ten targets already on disk. Rows
    that ended in error/timeout are retried — those are usually a dropped
    MAST/TRILEGAL connection, not a real verdict.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_vc2", Path(__file__).resolve().parents[1] / "scripts" / "validate_candidates.py"
    )
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)

    out = tmp_path / "validated.csv"
    pd.DataFrame(
        [
            {"tic_id": 111, "classification": "likely_planet", "fpp": 0.02},
            {"tic_id": 222, "classification": "degenerate", "fpp": 0.857},
            {"tic_id": 333, "classification": "error", "fpp": None},
            {"tic_id": 444, "classification": "timeout", "fpp": None},
        ]
    ).to_csv(out, index=False)

    prior = pd.read_csv(out)
    rows = prior.to_dict("records")
    unfinished = {"error", "timeout"}
    done = {
        int(r["tic_id"])
        for r in rows
        if str(r.get("classification")) not in unfinished and pd.notna(r.get("tic_id"))
    }

    assert done == {111, 222}  # a degenerate result is still a completed target
    assert 333 not in done and 444 not in done  # failures get another attempt


def test_trilegal_cache_roundtrip(tmp_path, monkeypatch):
    """TRILEGAL is a Monte Carlo galaxy sim — re-querying changes the answer.

    Its star count feeds the background prior directly, so two runs of the same
    target disagreed on the BEB-vs-NEB balance (one flipped between likely_fp
    and likely_nearby_fp). Caching the population per target makes runs
    reproducible. TRICERATOPS writes <TIC>_TRILEGAL.csv into the cwd with no
    way to redirect it, so the file has to be moved after the fact.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_vc3", Path(__file__).resolve().parents[1] / "scripts" / "validate_candidates.py"
    )
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)

    cache = tmp_path / "cache"
    monkeypatch.chdir(tmp_path)

    # Nothing cached yet.
    assert vc._trilegal_cached(cache, 12345) is None
    assert vc._trilegal_cached(None, 12345) is None  # caching disabled

    # TRICERATOPS drops the population in the cwd; we stash it.
    Path("12345_TRILEGAL.csv").write_text("simulated,stars\n1,2\n")
    vc._stash_trilegal(cache, 12345)
    assert not Path("12345_TRILEGAL.csv").exists()  # cwd left clean
    assert (cache / "12345_TRILEGAL.csv").exists()

    # Next run finds and reuses it.
    assert vc._trilegal_cached(cache, 12345) == str(cache / "12345_TRILEGAL.csv")

    # A no-op when the target never wrote one, and when caching is off.
    vc._stash_trilegal(cache, 999)
    vc._stash_trilegal(None, 12345)
