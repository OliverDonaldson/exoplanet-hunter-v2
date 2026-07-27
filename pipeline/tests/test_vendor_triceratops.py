"""The vendored TRICERATOPS fork keeps its patches.

The FPP/NFPP numbers this project publishes depend on the fixes in
`pipeline/vendor/triceratops` (see its README). A stock `pip install
triceratops` satisfies the same version constraint and silently produces
different — in one case constant — output, so the suite asserts the patched
behaviour rather than trusting the environment.

The fork's own tests run here too, as a subprocess so their sys.path
manipulation cannot leak into ours.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "triceratops"

triceratops = pytest.importorskip("triceratops", reason="pipeline[validation] not installed")


def test_installed_triceratops_is_the_vendored_fork():
    """A stock install satisfies `triceratops>=1.0` but lacks every patch."""
    import triceratops.triceratops as tri

    resolved = Path(inspect.getfile(tri)).resolve()
    assert VENDOR_ROOT in resolved.parents, (
        f"triceratops resolves to {resolved}, not the vendored fork at {VENDOR_ROOT}. "
        "Reinstall with: pip install -e pipeline/vendor/triceratops --no-deps"
    )


def test_evidence_integral_survives_underflow():
    """NC-01: mean(exp(lnL + 600)) underflows to -inf for lnL < -600.

    Long light curves reach lnL ~ -2000 easily, and an -inf evidence poisons
    the whole scenario table.
    """
    from triceratops._numerics import _log_mean_exp

    n = 50_000
    lnL = np.full(n, -1500.0)

    # The old scheme: precondition for this test to mean anything.
    assert not np.isfinite(np.log(np.mean(np.nan_to_num(np.exp(lnL + 600)))))

    assert _log_mean_exp(lnL, N_total=n) == pytest.approx(-1500.0, abs=1e-10)


def test_probability_normalisation_flags_degenerate_runs():
    """NC-01b: exp(lnZ)/sum(exp(lnZ)) is 0/0 = NaN when every lnZ underflows.

    That NaN reached us as `fpp=nan` classified "inconclusive" — a silent
    failure. The replacement reports a status instead.
    """
    from triceratops._numerics import _normalize_probabilities

    probs, status = _normalize_probabilities(np.array([-2000.0, -2001.0, -2002.0]))
    assert status == "ok"
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(probs))

    for lnZ, expected in (
        (np.full(5, -np.inf), "all_neginf"),
        (np.array([-1.0, np.nan, -3.0]), "anomaly"),
        (np.array([-1.0, np.inf, -3.0]), "anomaly"),
    ):
        probs, status = _normalize_probabilities(lnZ)
        assert status == expected
        assert not np.any(np.isnan(probs))


def test_background_prior_is_a_natural_log():
    """NC-03: log10 in a natural-log sum deflates the prior by ln(10).

    Understating background scenarios biases FPP/NFPP low — toward validating
    planets — which is the dangerous direction.
    """
    from triceratops.priors import lnprior_background

    n_comp, delta_mags = 100, np.array([3.0])
    seps, contrasts = np.array([0.5, 1.0, 2.0]), np.array([2.0, 4.0, 6.0])
    expected_sep = np.interp(delta_mags, contrasts, seps)[0]
    arg = (n_comp / 0.1) * (1 / 3600) ** 2 * expected_sep**2

    got = lnprior_background(n_comp, delta_mags, seps, contrasts)[0]
    assert got == pytest.approx(np.log(arg), rel=1e-12)
    assert got != pytest.approx(np.log10(arg), rel=1e-6)


def test_psf_integral_is_analytic():
    """NC-02: ndtr closed form, not a per-pixel dblquad of Gauss2D."""
    import triceratops.triceratops as tri

    src = inspect.getsource(tri.target.calc_depths)
    assert "ndtr(" in src
    assert "dblquad(" not in src
    assert "Gauss2D" not in src


@pytest.mark.parametrize(
    "test_file",
    [
        "test_log_mean_exp.py",
        "test_analytic_psf.py",
        "test_background_prior_log_base.py",
        "test_beb_collision_mask.py",
    ],
)
def test_fork_own_test_suite(test_file):
    """Run the fork's tests (incl. NC-04 collision masks) in a subprocess.

    They insert their own repo root on sys.path, which must not leak here.
    """
    path = VENDOR_ROOT / "tests" / test_file
    assert path.exists(), f"vendored test missing: {path}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=VENDOR_ROOT,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
