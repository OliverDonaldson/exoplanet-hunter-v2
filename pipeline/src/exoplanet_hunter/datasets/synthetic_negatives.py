"""Synthetic negatives: light curves that cannot contain a transit, by construction.

Observation baseline correlates with the label in the catalogue, and no
architecture reaches that, because in the training labels it is true. Destroying
the transit in a real curve makes the negative label correct however long the
star was observed, diluting the association without touching a real label.

Two constructions from Kepler's Robovetter work (Coughlin 2016): **inversion**,
reflecting flux about its median so a transit becomes an impossible brightening,
and **scrambling**, permuting segments so a periodic transit smears across
phase. One still holding its transit is a mislabelled positive, invisible in
training, so neither trusts its own mechanism — see `assert_transit_destroyed`.
Numbers: `docs/experiments/stage-08-labels-and-negatives.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from exoplanet_hunter.eval.observation_bias import BASELINE_DAYS, baseline_days
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: The two constructions. Named rather than boolean so a row records which one
#: produced it and a later analysis can split on it.
INVERT = "invert"
SCRAMBLE = "scramble"
KINDS = (INVERT, SCRAMBLE)

#: Detection threshold, in sigma, at the original ephemeris. A construction
#: that leaves a signal above this has left a *detectable* transit, which is
#: what makes a row a mislabelled positive — not the fraction of depth that
#: survives. 3 sigma is deliberately conservative: the shortlist operates far
#: above it, so a residual this small cannot be what the model keys on.
MAX_SURVIVING_SIGMA = 3.0

#: Fewest segments a scramble may use. Two is the minimum that permutes at all,
#: and a single segment is the identity — which returns the light curve
#: unchanged and labels it negative.
MIN_SEGMENTS = 2


def invert_flux(flux: np.ndarray) -> np.ndarray:
    """Reflect flux about its median, turning transits into brightenings.

    The median rather than the mean: a deep transit drags the mean down into the
    dip, so reflecting about it would leave part of the transit still pointing
    downwards. The median is unmoved by a signal occupying a few percent of the
    cadences, which is exactly what a transit is.
    """
    values = np.asarray(flux, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("cannot invert a light curve with no finite flux")
    return 2.0 * float(np.median(values[finite])) - values


def scramble_flux(
    time: np.ndarray, flux: np.ndarray, *, n_segments: int = 8, seed: int = 42
) -> np.ndarray:
    """Permute contiguous segments of flux in time, leaving timestamps in place.

    Segment boundaries are drawn at random interior positions rather than evenly.
    Even spacing makes every segment the same length, and a length that happens to
    be an integer multiple of the transit period *preserves phase* — the fold at the
    original ephemeris comes back unchanged and the "negative" still holds its
    transit. Random boundaries make that coincidence measure-zero.

    The permutation is also checked to derange at least one segment: the identity is
    a legal draw from `permutation` and returns the curve untouched.
    """
    t = np.asarray(time, dtype=float)
    values = np.asarray(flux, dtype=float)
    if len(t) != len(values):
        raise ValueError(f"{len(t)} timestamps but {len(values)} flux points")
    if n_segments < MIN_SEGMENTS:
        raise ValueError(
            f"a scramble needs at least {MIN_SEGMENTS} segments, got {n_segments} — "
            "one segment is the identity, which relabels the curve without changing it"
        )
    if len(values) < n_segments:
        raise ValueError(f"{len(values)} cadences cannot be cut into {n_segments} segments")

    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, len(values)), size=n_segments - 1, replace=False))
    segments = np.split(values, cuts)

    # Redrawn rather than accepted: `permutation` may return the identity, and
    # for a small n_segments that is not rare (1/8! is small, but 1/2! is a half).
    for _ in range(64):
        order = rng.permutation(n_segments)
        if not np.array_equal(order, np.arange(n_segments)):
            break
    else:  # pragma: no cover - unreachable for n_segments >= 2
        raise RuntimeError("could not draw a non-identity segment permutation")

    return np.concatenate([segments[i] for i in order])


def folded_depth(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float
) -> float:
    """Fractional depth at one ephemeris: 1 − (in-transit / out-of-transit) median.

    Medians rather than means on both sides, so a handful of outliers in a
    sparsely-sampled transit window cannot manufacture or erase a depth.
    """
    t = np.asarray(time, dtype=float)
    values = np.asarray(flux, dtype=float)
    if period <= 0 or duration <= 0:
        raise ValueError(f"a fold needs a positive period and duration, got {period}, {duration}")

    phase = np.abs(np.mod(t - t0 + 0.5 * period, period) - 0.5 * period)
    in_transit = (phase < 0.5 * duration) & np.isfinite(values)
    out_transit = (phase >= 0.5 * duration) & np.isfinite(values)
    if not in_transit.any() or not out_transit.any():
        raise ValueError("the fold puts no finite cadence on one side of the transit window")

    baseline = float(np.median(values[out_transit]))
    if baseline == 0.0:
        raise ValueError("out-of-transit median is zero; depth is undefined")
    return 1.0 - float(np.median(values[in_transit])) / baseline


def transit_significance(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float
) -> float:
    """Signed depth in units of its own standard error at one ephemeris.

    Positive is a dip, negative a brightening. The error combines the
    out-of-transit scatter over both sample sizes, `sd * sqrt(1/n_in + 1/n_out)`,
    which is the standard two-sample error on a difference of means — the
    in-transit window is the small sample and usually sets it.
    """
    t = np.asarray(time, dtype=float)
    values = np.asarray(flux, dtype=float)
    if period <= 0 or duration <= 0:
        raise ValueError(f"a fold needs a positive period and duration, got {period}, {duration}")

    phase = np.abs(np.mod(t - t0 + 0.5 * period, period) - 0.5 * period)
    in_transit = (phase < 0.5 * duration) & np.isfinite(values)
    out_transit = (phase >= 0.5 * duration) & np.isfinite(values)
    n_in, n_out = int(in_transit.sum()), int(out_transit.sum())
    if n_in < 2 or n_out < 2:
        raise ValueError(
            f"the fold leaves {n_in} in-transit and {n_out} out-of-transit cadence(s); "
            "a significance needs at least two of each"
        )

    baseline = float(np.median(values[out_transit]))
    if baseline == 0.0:
        raise ValueError("out-of-transit median is zero; depth is undefined")
    depth = 1.0 - float(np.median(values[in_transit])) / baseline
    scatter = float(np.std(values[out_transit], ddof=1)) / abs(baseline)
    error = scatter * np.sqrt(1.0 / n_in + 1.0 / n_out)
    if error == 0.0:
        raise ValueError("out-of-transit scatter is zero; a significance is undefined")
    return depth / error


def assert_transit_destroyed(
    time: np.ndarray,
    original: np.ndarray,
    constructed: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    *,
    max_sigma: float = MAX_SURVIVING_SIGMA,
) -> float:
    """Raise unless no transit is *detectable* in `constructed`. Returns its sigma.

    The mechanisms are sound in the abstract and this does not trust them. A
    scramble whose segments align with the period, an inversion of a curve whose
    median sits inside the transit, a curve short enough that one segment holds
    every transit — each looks like a synthetic negative and still carries its dip,
    which training accepts silently as a mislabelled positive.

    It tests a detection significance, not a fraction of the original depth. The
    fraction is the wrong statistic on a real curve: a catalogue transit is often
    only a few sigma to begin with, so it divides two noisy small numbers, and on
    the first real run it mis-rejected three of four scrambles. A curve whose
    original transit is itself undetectable is rejected — there is nothing to
    destroy, so the check could neither pass nor fail honestly.
    """
    before = transit_significance(time, original, period, t0, duration)
    if abs(before) <= max_sigma:
        raise ValueError(
            f"the original curve shows only {before:+.1f} sigma at this ephemeris "
            f"(threshold {max_sigma}), so there is no detectable transit to destroy and "
            "this check can neither pass nor fail — draw a host with a real transit"
        )
    after = transit_significance(time, constructed, period, t0, duration)
    if abs(after) > max_sigma:
        raise ValueError(
            f"the construction left a {after:+.1f} sigma signal at the original ephemeris "
            f"(limit {max_sigma}, original {before:+.1f}) — a transit is still detectable "
            "here, so this is a mislabelled positive and training cannot tell the difference"
        )
    return after


def make_synthetic_negative(
    time: np.ndarray, flux: np.ndarray, kind: str, *, n_segments: int = 8, seed: int = 42
) -> np.ndarray:
    """Dispatch to one construction. Unknown kinds raise rather than defaulting."""
    if kind == INVERT:
        return invert_flux(flux)
    if kind == SCRAMBLE:
        return scramble_flux(time, flux, n_segments=n_segments, seed=seed)
    raise ValueError(f"unknown synthetic-negative kind {kind!r}; expected one of {list(KINDS)}")


@dataclass(frozen=True)
class NegativeDraw:
    """Which hosts were drawn to become synthetic negatives, and what it cost."""

    hosts: pd.DataFrame
    n_requested: int
    #: Wasserstein-style summary: median baseline of the draw against the target.
    median_baseline: float
    target_median_baseline: float

    @property
    def n(self) -> int:
        return len(self.hosts)

    def report(self) -> str:
        return (
            f"{self.n} synthetic-negative hosts of {self.n_requested} requested; "
            f"median baseline {self.median_baseline:.0f} d against the positives' "
            f"{self.target_median_baseline:.0f} d"
        )


def draw_negative_hosts(
    candidates: pd.DataFrame, *, n: int, seed: int = 42, n_strata: int = 4
) -> NegativeDraw:
    """Draw hosts whose baseline distribution matches the **positives'**.

    This is the part that breaks the correlation rather than merely diluting it.
    Drawn uniformly, synthetic negatives would inherit the pool's own short-baseline
    bulk, making "short baseline" an even stronger negative cue and moving the
    correlation the wrong way. Matching the positives puts negatives exactly where
    the catalogue currently has almost none, which is where the confound lives.

    Raises when the pool cannot supply a stratum rather than backfilling: a quietly
    backfilled draw returns a clean number about a distribution never built.
    """
    required = {"tic_id", "label"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"negative-host candidates are missing {sorted(missing)}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    frame = candidates.drop_duplicates("tic_id").copy()
    if BASELINE_DAYS not in frame.columns:
        frame[BASELINE_DAYS] = baseline_days(frame)
    frame = frame[np.isfinite(frame[BASELINE_DAYS].to_numpy(dtype=float))]

    positives = frame[frame["label"] == 1]
    if positives.empty:
        raise ValueError("no positives to match the baseline distribution of")
    target_median = float(positives[BASELINE_DAYS].median())

    # Strata are cut on the POSITIVES' quantiles, not the pool's — the pool's
    # own quantiles describe the distribution being corrected, so matching to
    # them would reproduce it.
    edges = np.unique(
        np.quantile(positives[BASELINE_DAYS].to_numpy(float), np.linspace(0.0, 1.0, n_strata + 1))
    )
    if len(edges) < 2:
        raise ValueError("the positives' baselines have no spread to stratify on")
    edges[0], edges[-1] = -np.inf, np.inf

    rng = np.random.default_rng(seed)
    per_stratum = max(1, n // (len(edges) - 1))
    drawn = []
    for lo, hi in pairwise(edges):
        block = frame[(frame[BASELINE_DAYS] >= lo) & (frame[BASELINE_DAYS] < hi)]
        if len(block) < per_stratum:
            raise ValueError(
                f"baseline stratum [{lo:.0f}, {hi:.0f}) d holds {len(block)} eligible host(s) "
                f"but {per_stratum} are needed. Backfilling from the short-baseline bulk would "
                "return a draw that does not match the positives — reduce n instead"
            )
        drawn.append(block.sample(n=per_stratum, random_state=int(rng.integers(0, 2**31 - 1))))

    hosts = pd.concat(drawn, ignore_index=True)
    draw = NegativeDraw(
        hosts=hosts,
        n_requested=n,
        median_baseline=float(hosts[BASELINE_DAYS].median()),
        target_median_baseline=target_median,
    )
    log.info("[synthetic-negatives] %s", draw.report())
    return draw
