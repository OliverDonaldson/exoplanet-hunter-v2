"""Synthetic negatives: light curves that cannot contain a transit, by construction.

**What this is for.** Observation baseline correlates +0.3874 with the label on
TESS and the branch model reads it at +0.5155 — *above* the labels it learned
from. The mechanism is confirmation bias in the catalogue: a target observed
across many sectors accumulates the follow-up that promotes it to confirmed,
while a briefly-observed one stays a candidate or is retired. No architecture
reaches that, because in the training labels it is true.

A synthetic negative breaks it at the source. Take a real light curve, destroy
any coherent transit in it, and label the result negative. The label is now
correct **regardless of how long the star was observed**, so a negative drawn
from a long-baseline host carries the opposite of the catalogue's association.
Drawn to match the positives' baseline distribution, they dilute the correlation
without touching a single real label — which is the only one of stage 8's three
interventions with no external dependency and the cleanest causal story.

**Two constructions, both standard.** Kepler's Robovetter was characterised
against exactly these (Coughlin 2016, KSCI-19114):

- **inversion** — reflect the flux about its median, so transits become
  brightenings. An astrophysical transit cannot go up, so anything recovered
  from an inverted curve is a false alarm by definition.
- **scrambling** — cut the curve into segments and permute them in time while
  leaving the timestamps in place. A periodic transit folded at its original
  ephemeris now mixes phases and smears out.

Both preserve the star's own noise, cadence gaps and systematics, which a
generated curve would not. That is the whole point: the negative has to be hard
in the same way real negatives are hard, or the model learns to spot the
synthesis instead of the absence of a transit.

**The failure mode this module is built around.** A "synthetic negative" that
still contains its transit is a *mislabelled positive*, and it is invisible —
training accepts it, the loss barely moves, and the intervention appears to have
been tried. So the constructions do not trust their own mechanism: they measure
the residual transit depth at the original ephemeris and **raise** when it has
not collapsed. See `assert_transit_destroyed`.
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

#: Fraction of the original injected depth that may survive the construction.
#: A transit reduced to under a twentieth of itself is below this catalogue's
#: shallowest real signals; anything more and the "negative" still carries the
#: thing it is supposed to lack.
MAX_SURVIVING_DEPTH_FRACTION = 0.05

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

    Segment boundaries are drawn at random interior positions rather than at
    even spacing. Even spacing makes every segment the same length, and a
    segment length that happens to be an integer multiple of the transit period
    **preserves phase** — the fold at the original ephemeris comes back
    unchanged and the "negative" still contains its transit. Random boundaries
    make that coincidence measure-zero instead of a property of the grid.

    The permutation is also checked to be a derangement of at least one segment;
    the identity is a legal draw from `permutation` and returns the curve
    untouched.
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


def assert_transit_destroyed(
    time: np.ndarray,
    original: np.ndarray,
    constructed: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    *,
    max_fraction: float = MAX_SURVIVING_DEPTH_FRACTION,
) -> float:
    """Raise unless the construction actually removed the signal. Returns the ratio.

    The mechanisms are sound in the abstract and this does not trust them. A
    scramble whose segments align with the period, an inversion of a curve whose
    median sits inside the transit, a curve short enough that one segment holds
    every transit — each returns something that looks like a synthetic negative
    and still carries its dip. Training would accept it silently as a
    mislabelled positive, the loss would barely move, and the intervention would
    be recorded as tried.
    """
    before = abs(folded_depth(time, original, period, t0, duration))
    if before <= 0.0:
        raise ValueError(
            "the original curve has no depth at this ephemeris, so there is nothing to "
            "destroy and the check cannot pass or fail — build the negative from a host "
            "with a real or injected transit"
        )
    after = abs(folded_depth(time, constructed, period, t0, duration))
    ratio = after / before
    if ratio > max_fraction:
        raise ValueError(
            f"the construction left {ratio:.1%} of the transit depth at the original "
            f"ephemeris (limit {max_fraction:.0%}) — this is a mislabelled positive, not a "
            "synthetic negative, and training cannot tell the difference"
        )
    return ratio


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
    Synthetic negatives drawn uniformly would inherit the pool's own baseline
    distribution, which is dominated by short-baseline targets — adding them
    makes "short baseline" an even stronger negative cue and moves the
    correlation the wrong way. Matching the positives instead puts negatives at
    exactly the baselines where the catalogue currently has almost none, which is
    where the confound lives.

    Raises when the pool cannot supply a stratum, rather than backfilling from
    the short-baseline bulk: a quietly backfilled draw returns a clean number
    about a distribution that was never built.
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
