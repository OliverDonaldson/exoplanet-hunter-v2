# Vendored TRICERATOPS (patched)

Upstream: [stevengiacalone/triceratops](https://github.com/stevengiacalone/triceratops)
v1.0.20, MIT licensed (`LICENSE` is upstream's, unmodified). Vendored as
`1.0.20+exohunter.1` so `pip list` never implies this is the stock release.

We vendor rather than pin because the FPP/NFPP numbers this project reports
depend on the fixes below, and a stock `pip install triceratops` silently
produces different — in one case constant — output.

## Why it is patched

Stage 2 of the candidate shortlist returned **FPP = 0.75 and NFPP = 0.0 for
all 20 targets**, across S/N 3.1–84.5 and 17–1,372 nearby stars. That is not a
result, it is a degenerate run: with the whole scenario table at a uniform
1/12, `FPP = 1 - 3/12 = 0.75` exactly and NFPP falls to a hardcoded branch.
Fixing the depth units on our side (`edd3715`) made the scenarios compute, and
the next target returned **FPP = NaN** — the stock normalisation is
`exp(lnZ)/Σexp(lnZ)`, which underflows to 0/0 on long light curves.

## The patches

| # | file | change |
|---|---|---|
| NC-01 | `_numerics.py` (new), `marginal_likelihoods.py` | `_log_mean_exp` — logsumexp evidence integral replacing `mean(exp(lnL + 600))`, which underflows to `-inf` for `lnL < -600`. 22 call sites. `-inf`/NaN draws contribute zero weight but still count in the denominator; `N_total` is a required keyword so a caller cannot silently pass `len(finite)` and overestimate evidence. |
| NC-01b | `_numerics.py`, `triceratops.py` | `_normalize_probabilities` — logsumexp normalisation replacing `exp(lnZ)/Σexp(lnZ)`, plus a `FPP_degenerate` flag and `RuntimeWarning` distinguishing "every draw geometrically invalid" from "NaN/+inf anomaly". `self.lnZ` is retained for diagnostics. |
| NC-02 | `triceratops.py`, `funcs.py` | Analytic PSF integral: `scipy.special.ndtr` closed form replacing a per-pixel `scipy.integrate.dblquad` of `Gauss2D`. Exact, not an approximation, and far faster — `calc_depths` integrates over every aperture pixel for every star. |
| NC-03 | `priors.py` | `lnprior_background` used `np.log10` where the value is added to natural-log likelihoods, deflating the background-star prior by a factor of ln(10) ≈ 2.303. Understating background scenarios biases FPP/NFPP **low**, i.e. toward validating planets. |
| NC-04 | `marginal_likelihoods.py` | Collision-mask fixes in the parallel paths of the EB scenarios: `lnZ_BEB`'s `q < 0.95` branch used `coll_twin` (less restrictive, since `a_twin > a`, admitting physically impossible configurations) and `lnZ_NEB_unknown`'s `q >= 0.95` twin branch used `coll`. Both serial paths were already correct; the parallel paths now match. |
| NC-05 | `likelihoods.py` | Compatibility shims so pytransit 2.2.0 imports on this env: `np.int` (gone in NumPy 1.24), `np.trapz` (NumPy 2.0), `scipy.integrate.trapz` (SciPy 1.14) and a `pkg_resources` stub (setuptools 81). The fork shipped only the two NumPy shims, so the package could not be imported without our `validation/statistical.py::_install_triceratops_compat_shims` running first — which defeats vendoring. That helper stays as belt-and-braces for stock installs. |

## Tests

`tests/` are the fork's own, run by our suite via `pipeline/tests/test_vendor_triceratops.py`.
Each asserts the *old* behaviour fails and the new one is correct — e.g.
`test_log_mean_exp.py::test_regression_old_fails_new_passes` first proves the
`+600` scheme underflows at `lnL = -1500`, then that the replacement returns
exactly `-1500`.

## Updating

Upstream is unlikely to carry these patches; re-apply them onto any newer
release rather than dropping them. `calc_depths` expects `tdepth` as a
**fraction** despite its docstring saying ppm (it computes `tdepth/fluxratio`
and zeroes anything `> 1`) — that is upstream behaviour we deliberately did
*not* change, so our caller converts. See
`validation/statistical.py::validate_target`.
