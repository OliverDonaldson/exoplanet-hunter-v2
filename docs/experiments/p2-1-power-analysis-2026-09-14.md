# P2.1 — the power analysis, and the metric the gate will use · 2026-09-14

`PLAN.md` §5 item P2.1, and part C of [#78](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/78).
Its exit criterion: *"An experiment file naming the metric the gate will use and
the minimum detectable effect at the current n."* Both are below.

The standing recommendation was carried into this as a **proposal to test, not a
decision** — the maintainer confirmed that framing on 2026-09-14. It is
**half accepted and half rejected**, and the reason for the rejection is not the
one anybody expected.

Reproduction: `python pipeline/scripts/power_analysis.py --n-boot 2000 --n-reps 25`,
tracked in the repository. Arms are `models/phase1/arm-c-control` and
`arm-d-difference`, inner-joined on `tic_id` so every bootstrap draw is paired.
Nothing was trained and nothing was promoted.

## 1. Stability, and the thing the record had not measured

TESS, the gating slice: n = 2,367, 1,300 positive, 1,067 negative. Eleven
negatives sit above the 1% FPR cut, nine on `dv_usable` — the record's
"eight-to-ten row statistic", confirmed.

| metric | arm C | arm D | D−C | bootstrap sd | **seed sd** | sd @5 members | **MDE @5** |
|---|---:|---:|---:|---:|---:|---:|---:|
| recall @1% FPR | 0.2354 | 0.2208 | −0.0146 | 0.0383 | 0.0467 | 0.0436 | **0.1222** |
| recall @5% FPR | 0.6008 | 0.5677 | −0.0331 | 0.0249 | 0.0520 | 0.0341 | 0.0954 |
| pAUC FPR≤0.1 | 0.7541 | 0.7485 | −0.0056 | 0.0071 | 0.0230 | 0.0125 | 0.0351 |
| ROC-AUC | 0.9176 | 0.9160 | −0.0016 | 0.0021 | 0.0093 | 0.0047 | **0.0131** |
| PR-AUC | 0.9201 | 0.9205 | 0.0004 | 0.0033 | 0.0122 | 0.0064 | 0.0179 |

The bootstrap sd for recall @1% FPR is **0.0383**, reproducing the 0.0437 in
`known-limits.md` to within its own bootstrap error. That much was known.

**What was not: seed sd exceeds bootstrap sd for every metric.** ROC-AUC's
seed-to-seed spread across three members of one arm — identical data, identical
architecture — is **0.0093 against a sampling sd of 0.0021**, a factor of 4.4.
For pAUC it is 0.0230 against 0.0071, a factor of 3.2.

Two consequences, both of which change P2.2:

- **A paired bootstrap alone under-states the uncertainty a gate faces**, because
  it measures the smaller of the two components. P2.2 as written — *"the gate
  reads a confidence interval… paired bootstrap on the contrast"* — is necessary
  and not sufficient. The interval has to carry the seed term too.
- **The more sampling-stable the metric, the more members it needs before rows
  become the binding constraint.** Members required for the seed term to fall
  below the sampling term: **~20 for ROC-AUC, ~11 for pAUC, ~2 for recall @1%
  FPR**. Choosing a stable metric moves the bottleneck from n to member count.
  At the 5 members/fold planned for the clean run, ROC-AUC's seed term is still
  the larger half of its total.

## 2. Power — which metric detects a degradation it is shown

Stability alone cannot choose a metric: they are on different scales, so a
smaller sd is not by itself more power. The ranking is degraded by a controlled
amount (rank-space Gaussian jitter, width `w`) and each metric is scored by
z = mean |change| / bootstrap sd of that change, averaged over 25 realisations.
1.96 is significance.

**Degradation applied to the whole ranking:**

| metric | w=0.02 | w=0.05 | w=0.1 | w=0.2 |
|---|---:|---:|---:|---:|
| recall @1% FPR | 0.58 | 0.74 | 0.97 | 2.21 |
| recall @5% FPR | 0.44 | 0.97 | 2.53 | 6.04 |
| pAUC FPR≤0.1 | 0.94 | 1.80 | 3.31 | 6.88 |
| **ROC-AUC** | **1.42** | **3.53** | **6.25** | **11.19** |
| PR-AUC | 1.01 | 1.66 | 3.86 | 8.64 |

ROC-AUC dominates at every level. recall @1% FPR is last at every level and does
not reach significance until `w = 0.2`, which is a gross degradation of the
whole ranking.

**Degradation confined to the top of the ranking** — the case pAUC exists for,
and the region the product's job lives in — at `w = 0.1`:

| metric | top 5% | top 10% | top 20% |
|---|---:|---:|---:|
| recall @1% FPR | 0.97 | 1.09 | 0.58 |
| recall @5% FPR | 0.00 | 0.33 | 1.15 |
| pAUC FPR≤0.1 | 0.84 | 0.98 | **1.36** |
| ROC-AUC | 0.81 | 0.92 | 1.27 |
| PR-AUC | 0.62 | 0.99 | 1.04 |

**Nothing crosses 1.96. Not one metric, at any top fraction.**

*(Methodological note, recorded because it changed an answer: a first pass used
a single realisation of the jitter per cell and produced z values that reordered
the metrics between runs — ROC-AUC read 1.82 at top-20% on one draw and 0.52 on
another. The z is itself a noisy statistic. Every number above is the mean of 25
realisations; the single-draw version is not reliable and is not reported.)*

## 3. What this decides

**Accepted: recall @1% FPR is demoted from gating.** It is the weakest metric
under uniform degradation at every level tested, its MDE at 5 members is
**0.1222** — larger than the entire difference between the best and worst arms
the record contains — and it is cut at nine to eleven rows. It stays reported,
because it is the shortlist criterion the console describes and a reader is
entitled to it. It never decides a promotion again.

**Rejected: pAUC over FPR ∈ [0, 0.1] as the gating metric.** The proposal was
sound in motivation and does not survive measurement. ROC-AUC detects the same
degradation with **1.9x** pAUC's z under uniform degradation (6.25 against 3.31
at w = 0.1), and on localised degradation — the case pAUC was chosen for — the
two are within noise of each other and *both* are insignificant. pAUC buys
relevance that cannot be cashed at this n.

**Decided: the gate reads ROC-AUC, with pAUC as a veto.**

- **ROC-AUC decides.** Most powerful at every point measured; MDE **0.0131** on
  TESS at 5 members per fold.
- **pAUC FPR≤0.1 may veto but never promote.** A candidate that improves ROC-AUC
  while regressing pAUC beyond pAUC's own interval is improving the bulk
  ordering at the follow-up region's expense, which is the failure mode the
  standing recommendation was written to prevent. Using it as a one-sided guard
  keeps that protection without paying the power penalty for it.
- **recall @1% FPR and PR-AUC are reported, and gate nothing.**
- **The interval both are read against must include the seed term**, per §1.

## 4. The finding that outranks the metric choice

**No metric available to this project can detect a change confined to the
follow-up region.** Top 5%: best z = 0.97. Top 10%: 1.09. Top 20%: 1.36.

This is not a metric-selection problem and no choice in §3 fixes it. It is a
statement about how few labelled rows land in the top of the ranking: the region
the console exists to order is the region the evidence base cannot see. Every
architecture comparison in the record was, on this axis, unable to answer the
question it was asked.

The route out is not a better statistic over the same labels. It is **P2.4,
injection-recovery at scale**, which constructs its positives at controlled S/N
and therefore sets its own n in exactly the region the labels are thinnest. That
reorders §5: P2.4 stops being the fourth item and becomes the one that decides
whether the follow-up region is measurable at all. Recorded as a decision rather
than a silent re-plan, per #78's instruction.

**Injection completeness above the null floor is therefore kept in the gate**, as
the standing recommendation proposed — but it cannot be specified here, because
its own power has never been measured. P2.4 must report the completeness curve's
standard error per S/N bin before completeness can gate anything, and P2.1
cannot name a minimum detectable effect for a measurement that does not yet
exist.

## 5. Limits of this analysis

- Three members per arm. The seed sd in §1 is estimated from **three draws**, the
  same thinness `known-limits.md` already carries against every floor in the
  record. The 5-member and 20-member projections assume the seed term shrinks as
  1/√M, which holds for independent members and is not verified here.
- One contrast, C″ against D″, on one shard set. Whether the seed/sampling ratio
  holds for other architecture pairs is untested.
- The degradation is synthetic. It is monotone and uniform in rank space, which
  a real architecture change is not; it measures a metric's *sensitivity*, not
  its agreement with any particular improvement.
- pAUC is McClish-standardised. The ranking by stability does not depend on that
  choice, but the absolute values do.

---

## Note appended 2026-09-17 — the seed sd was one arm's, on 2 df, and the MDE carried one seed term

This entry is not edited. The two corrections below change §1's MDE column and
§3's headline figure; everything else in this file stands, including the metric
ranking, the demotion of recall @1% FPR, and §4's finding about the follow-up
region. Measured by `power_analysis.py --n-boot 2000`, which now runs the census
described here. Nothing was trained. Issue [#93](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/93).

**1. The seed sd came from one arm's three members.** `seed_spread` read
`arm C` only, so §1's 0.0093 is a 2-df estimate. §5 said so; what it could not
say is how unstable that is. Fifteen multi-member runs were already on disk. Per-run
ROC-AUC seed sd across them spans **0.0032 to 0.0193, a 6x range**, which is what
2 df looks like. Pooled on the TESS gating slice:

| architecture | pooled seed sd, ROC-AUC | runs | df |
|---|---:|---:|---:|
| `cnn_branches` | **0.0108** | 13 | 26 |
| `cnn_dualview` | **0.0062** | 2 | 4 |
| all pooled | 0.0103 | 15 | 30 |

Arms C'' and D'' are `cnn_branches`. Their sd was being read as the project's
MDE for every comparison, including P2.6's, which compares both architectures —
the category error `docs/index.md` rule 7 forbids. Decision
[#34](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/34) judged
that "roughly ten draws would" retire the thin-floor limitation and that the
research was frozen; thirteen extra draws did not need any.

**2. The MDE carried one seed term where a contrast needs two.** §1 computed
`hypot(boot, sd_seed / sqrt(M))`, the sd of one arm's mean.
`promotion.py::decision_floor` has read
`sqrt(sd_cand^2/n_cand + sd_inc^2/n_inc)` since 4.1b, and 4.1b argues for it
explicitly. The gate and this analysis disagreed about the same quantity.

Corrected, at 5 members per fold, TESS ROC-AUC, 80% power:

| comparison | M=3 | M=5 | M=10 |
|---|---:|---:|---:|
| branch vs branch | 0.0255 | **0.0201** | 0.0148 |
| dual-view vs dual-view | 0.0155 | **0.0126** | 0.0098 |
| *as published above* | 0.0162 | *0.0131* | 0.0101 |

**So 0.0131 is about right for a dual-view challenger and too lenient by 1.5x
for a branch one.** `power_analysis.py` now prints both columns, `MDE P2.1` and
`MDE 2-arm`, so the published figure stays locatable beside the correct one.

**3. Pairing on member index does not help, and that is worth recording.** If
the two arms' seed draws were shared, the contrast would need no second term.
Measured: the member-paired contrast sd is **0.0175**, against
`sqrt(2) x 0.0093 = 0.0130` predicted under independence, and the member main
effect common to both arms is ~0. The draws are independent or worse, so both
terms stand. This rules out the obvious fix before compute is spent on it.

**4. What §3's decision looks like under the correction.** The gate still reads
ROC-AUC with pAUC as a one-sided veto; nothing about the metric choice changes.
But the veto's own bar moves: dual-view pAUC seed sd is 0.0171, so at 5 members
it can only fire on a regression beyond **+-0.036**. A veto with that trigger is
close to inert on a metric whose range of interest is smaller than its threshold.
pAUC stays reported; it should not be relied on as the relevance guard. Recorded
as [#105](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/105).
