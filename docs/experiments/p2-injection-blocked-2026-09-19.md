# P2 — the injection instrument read as a blocked design · 2026-09-19

A re-analysis of `results/injection_recovery.parquet` as it stands. **Nothing was
injected, trained or promoted**; the 990 rows are the ones already on disk, from
the champion `ca906040`.

Reproduction: `python pipeline/scripts/injection_recovery.py --analyse`.

## On pre-registration

This corrects an analysis rather than fixing the reading of a future run, so
rule 6 does not apply in its usual form, and the honest account of the order is:
the block-level lift table below was computed first, during the review; the
estimand and the clustered inference were then fixed in a comment on
[#97](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/97) before
the GEE was fitted. Nothing was chosen after seeing which specification gave the
nicer number, but this is not a pre-registered result and should not be read as
one.

## 1. What was wrong

#97 proposes three changes, and the third contradicts the first as written.

It reports the dose-response GLM as `beta = +3.754, se 0.285, z = 13.2 on 946
residual df`, and, two paragraphs later, a measured **1.57x variance reduction
from blocking on host**. Those cannot both hold: a 1.57x blocking gain *is*
intra-host correlation, and 946 residual df asserts there is none.

It also reports **S/N at 50% completeness = 19.4** from a two-parameter logistic.
That model forces `P -> 0` as `S/N -> 0`. Measured, the null-injection floor is
**0.264** overall — 0.123 on the 24 false-positive hosts, 0.467 on the 16 planet
hosts. A logistic with no floor term cannot represent that and is dragged toward
it; fitting one here returns S/N50 between 2.3 and 7.7 depending on the
specification, none of which reproduces 19.4.

## 2. The primary result: within-host lift

Host is the block, its own `S/N = 0` cell is the control, and each block is
collapsed to one value before the test — which is what makes `G - 1` the honest
degrees of freedom. This is the paired-difference test STAT 292 Part 1 names as
the two-treatment special case of a randomised block design.

| S/N | all 40 | t(39) | p | FP hosts (24) | planet hosts (16) | p (planet) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | −0.013 | −0.23 | 0.82 | −0.007 | −0.021 | 0.79 |
| 5 | +0.013 | 0.23 | 0.82 | +0.007 | +0.021 | 0.82 |
| 7 | +0.062 | 1.04 | 0.31 | +0.104 | **+0.000** | **1.00** |
| 10 | **+0.204** | **3.13** | **0.003** | +0.271 | +0.104 | 0.26 |
| 15 | +0.358 | 5.79 | 1e−6 | +0.451 | **+0.219** | **0.022** |
| 20 | +0.467 | 7.35 | 7e−9 | +0.542 | +0.354 | 0.003 |
| 30 | +0.583 | 8.89 | 6e−11 | +0.639 | +0.500 | 2e−4 |
| 50 | +0.671 | 10.94 | 2e−13 | +0.757 | +0.542 | 4e−5 |

**The served champion has no measurable response to an injected transit below
S/N 10, and none below S/N 15 on planet hosts.** Injecting a S/N = 7 transit
into a planet host moves its recovery rate by exactly zero.

That is [#78](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/78)'s
thesis — "the score does not track transit evidence" — with a test attached, and
it is a stronger statement than any completeness curve.

## 3. S/N at half the maximum lift

With a floor, two estimands diverge and have to be named:

- **half of the achievable lift**, which the design measures directly
- **absolute 50% recovery**, which is reached at a lift of `0.5 - floor`

With floor 0.264 and maximum lift 0.671, absolute-50% needs a lift of 0.236 —
only 35% of what is achievable. Interpolating the block-level lifts in
log10(S/N):

| population | S/N at half-max lift |
|---|---:|
| all hosts | **14.1** |
| false-positive hosts | 12.7 |
| planet hosts | **16.8** |

#97's 19.4 is the right order and closest to the planet-host figure; the
floorless fits are the ones that were wrong.

## 4. The dose-response, with the clustering it has

`beta` on log10(S/N), 880 rows with S/N > 0 in 40 blocks, controlling for period
and host class:

| model | beta | se | statistic | df |
|---|---:|---:|---:|---:|
| independent rows, as #97 reports | +3.384 | 0.253 | z = 13.4 | 875 |
| cluster-robust by host | +3.384 | 0.417 | **t = 8.1** | **39** |
| GEE, exchangeable within host | +3.223 | 0.418 | t = 7.7 | 39 |

**SE inflation 1.65x**, so the effective sample size is about **322 of 880**, and
the GEE's within-host working correlation is **0.372**.

The finding survives comfortably. The *precision* does not, and the MDE #97
derives from it inherits the correction: the claimed ~0.009 completeness becomes
roughly 0.011. That is still the smallest detectable effect any instrument in
this project has in the follow-up region, which is why #97 stays priority-
promoted — the conclusion is unchanged and only the number moves.

A caution on converting that inflation into an intra-cluster correlation: the
textbook `DEFF = 1 + (m-1)rho` describes a cluster-level *mean*, and S/N varies
*within* host, so the slope is partly estimated within-cluster and is far less
affected than that formula implies. An earlier note in this review quoted
`rho = 0.091` from exactly that back-calculation and it was wrong. The two
defensible numbers are the measured SE inflation and the GEE's working
correlation, reported above.

## 5. The host-class confound

The two host classes are different experiments, not a nuisance split. FP hosts
start at a floor of 0.123 and reach +0.757 lift; planet hosts start at 0.467 and
saturate at +0.542. One curve has a ceiling the other does not. Pooling them
averages into a shape neither has, and the pooled floor correction already in
`report()` — `(fraction - baseline) / (1 - baseline)` over the *pooled* control
mean — cannot represent it.

On the logit scale the planet-host coefficient is **+0.386 (se 0.521)** once S/N
and period are controlled, so the classes differ mainly in baseline and ceiling
rather than in slope.

## 6. What this changes

- The completeness headline is the **within-host lift**, floor-corrected by
  construction, not the raw recovery fraction and not a floorless logistic.
- Inference on this artefact is on **39 df**, not ~840.
- Strata are reported separately, always.
- `--analyse` makes all of it reproducible from the parquet without scoring.

## 7. What it does not establish

The 990 rows come from one model, `ca906040`, because until
[#112](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/112) the
harness could score only the registry champion. No arm has been read on this
instrument, and a multi-member arm still needs
[#113](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/113). The
grid also spends three of its eight graded levels in the region where the
response is indistinguishable from zero
([#116](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/116)).

Nothing here is a completeness figure for any model other than the one currently
served.
