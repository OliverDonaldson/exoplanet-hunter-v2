> Written 2026-09-10. The primary hypothesis below was stated by the person
> commissioning the audit *before* any of it was measured, and is recorded here
> with its own falsification. Frozen on the same rule as every other file here:
> a correction is a dated note appended under the entry it corrects, never an
> edit.

# Benchmark comparability — the Kepler inflation hypothesis is FALSIFIED, and only two of six comparisons survive (2026-09-10)

A citation audit ran on 2026-09-07 and its fixes are in
[PR #61](https://github.com/OliverDonaldson/exoplanet-hunter/pull/61); every
bibliographic locator in this repository is verified. This audit asks a
different question and re-verifies no citation: **are our numbers comparable to
the ones we cite, and where we imply they are, are we right?**

Five primary sources were read by one agent each. Six candidate metric pairs
were then judged. The verdicts and their adversarial refutation were to be run
by a second fleet; that fleet died on a session limit, so **the verdicts in §4
were reached by hand in the main session from the extracted evidence, and have
NOT been through the three-refuter panel this audit specified.** That is
recorded as a limitation of this file, not papered over — see §8.

## 1. What was tested

**The primary hypothesis, as stated at commission:**

> Our Kepler ROC-AUC of 0.9914 is inflated relative to the published Kepler
> numbers because restricting negatives to `koi_score < 0.5` keeps only the
> confidently-rejected false positives and discards the ambiguous ones, making
> the discrimination task easier than the one those papers solved.

It has two separable parts, and they do not stand or fall together:

- **the mechanism** — that the `koi_score < 0.5` threshold is what removes the
  ambiguous false positives;
- **the conclusion** — that our Kepler task is easier than the published one.

The mechanism is false. The conclusion is true, for reasons the hypothesis does
not name.

## 2. The mechanism is FALSIFIED — the score threshold is nearly inert

Measured against the NASA Exoplanet Archive TAP service on 2026-09-10, joining
`q1_q17_dr25_koi` and `cumulative` to `data/tables/labels/labels.parquet` and
`models/cv/ca906040.../predictions.parquet`.

The cut retains **3,813 of 4,839** cumulative false positives — 78.8%, which
confirms the report's "about 79%". The report's account of *why* the other
21.2% leave is correct ("dropped as disputed or unvetted"). The hypothesis's
account is not. The 1,026 discarded rows decompose as:

| why a cumulative FP is discarded | n | share |
|---|---:|---:|
| **absent from the DR25 table altogether** — never uniformly vetted | 893 | 87.0% |
| **DR25 dispositions it CANDIDATE** — DR25 disputes the FP call | 129 | 12.6% |
| **DR25 FALSE POSITIVE with `koi_score` ≥ 0.5** — failed the threshold | **4** | **0.4%** |

**Four KOIs.** The threshold that the hypothesis holds responsible for the
inflation removes four rows. The attrition is DR25 table membership, and the
report already says so.

The reason is structural: `koi_score` is near-degenerate on the class it is
applied to. Of 3,965 DR25 FALSE POSITIVE KOIs, 3,531 sit in [0, 0.001] and only
0.1% reach 0.5. The Robovetter does not hedge on the signals it rejects; it
hedges on the ones it passes, and those are dispositioned CANDIDATE, where
1,131 of 1,358 score above 0.75. **A `< 0.5` cut on a distribution that is 89%
at zero cannot be doing the work the hypothesis assigns it.**

Our own 1,250 labelled Kepler negatives inherit that: mean `koi_score` 0.0121,
median 0.000, 75th percentile 0.000, maximum 0.483.

### What the threshold *is* worth, quantified

The ambiguous negatives that do survive are genuinely harder, and the effect is
monotone. Out-of-fold Kepler AUC, all 1,245 positives against each band of
negatives:

| negative `koi_score` band | n | AUC | mean predicted prob |
|---|---:|---:|---:|
| [0, 0.001) | 879 | 0.9924 | 0.078 |
| [0.001, 0.1) | 72 | 0.9893 | 0.104 |
| [0.1, 0.25) | 23 | 0.9840 | 0.091 |
| [0.25, 0.5) | 18 | **0.9598** | 0.241 |

So the *direction* of the hypothesis is right — ambiguity costs AUC — but the
population it would cost us is not there to lose.

### The counterfactual Kepler AUC

The discarded rows were never scored by the served model, so the unrestricted
AUC cannot be computed. It can be bounded by resampling: enlarge the negative
class by 267 rows (21.2% of the enlarged class) drawn with replacement from a
donor band, 400 draws.

| donor behaviour for the restored rows | Kepler AUC | drop |
|---|---:|---:|
| as measured, restricted (baseline) | **0.9914** | — |
| like the `koi_score` ≥ 0.1 negatives | 0.9876 ± 0.0009 | 0.0038 |
| like the hardest retained band (≥ 0.25) | 0.9847 ± 0.0010 | 0.0067 |
| like confirmed planets (total contamination floor) | 0.8872 | 0.1042 |

**Without the restriction our Kepler AUC would plausibly be 0.985–0.988.** The
inflation attributable to the negative-class rule is on the order of
**0.004–0.007 AUC** — an order of magnitude too small to account for the
0.081 Kepler–TESS gap §6.4 calls the headline finding of the evaluation. The
0.887 row is a floor, not an estimate: it assumes every restored row scores like
a planet, which nothing supports.

## 3. What does make Kepler easy — three findings the hypothesis missed

**(a) Both classes are drawn from the Robovetter's confident ends.** Our Kepler
positives have median `koi_score` 1.000; our negatives, 0.000. The task is not
"planet or false positive" but "did the Robovetter say yes or no", and it said
so almost deterministically. All 1,977 cumulative CANDIDATE rows — where the
ambiguity actually lives — are held out by the label rules in §2.2 and appear
in neither class. **This, not the `< 0.5` threshold, is the real restriction.**
Islam does the same thing and reports 0.955, so it is not sufficient on its own
either; see §4.

**(b) The negative class is a subsample, not a census.** `catalog.py` calls
`_stable_sample` to take exactly 1,250 negatives from the 3,813-strong eligible
pool and exactly 1,250 positives from 2,748 cumulative CONFIRMED. The report
does not say this anywhere. The sampling itself is unbiased with respect to
anything astrophysical — `_stable_sample` ranks by `md5(f"{seed}:{tic_id}")`,
which is independent of SNR, depth and period by construction — so this is not
a hidden easiness. It is a **power** finding: the Kepler evaluation is 45% of
the available labelled Kepler population, and the report presents 2,500 as
though it were the dataset rather than a draw from it.

**(c) The CV grouping is inert on this labelled set.** `train.py` splits with
`StratifiedGroupKFold(groups=tic_id)`, and §5.1 justifies it by saying
"multi-planet systems and re-observed targets contribute several rows that share
a star, and a star split across folds leaks". On `labels.parquet` as it stands,
**every group has size 1** — 5,812 rows on 5,812 unique `tic_id`, group-size
distribution `{1: 5812}`. Grouped and ungrouped k-fold are the same partition
here.

This is **not** a leakage finding. Our leakage protection is real and, if
anything, stronger than grouping: the label builder emits one signal per host
star, so no sibling exists anywhere in the dataset to leak. But the protection
comes from **deduplication in the label builder**, not from the CV grouping,
and §5.1 credits the wrong mechanism. It also has a cost the report never
states: our Kepler set holds 2,500 of 8,054 DR25 KOIs, having dropped every
additional planet in every multi-planet system.

## 4. The comparability table (deliverable 1)

Sources as established by the reading agents. Where a paper is silent, that is
recorded as silent and not inferred.

| our metric | published metric it would be compared to | verdict | reason and evidence |
|---|---|---|---|
| Kepler ROC-AUC **0.9914 ± 0.0028**, n=2,237, balanced 1,245/992 | **ExoMiner** (Valizadegan 2022) — headline is *recall 0.936 at 99% precision* on 30,609 TCEs, PC 2,291 / AFP 3,538 / NTP 24,779, 10-fold CV grouped by target star | **NOT COMPARABLE** | Not a metric mismatch — a **missing metric**. The reader could not establish any ROC-AUC for Valizadegan 2022 from the primary source; its headline is recall at fixed precision. There is nothing to place 0.9914 beside. Separately, their evaluation population is 92.5% negative against our 44%, and ROC-AUC on a 1:12 prior is not the same quantity as ROC-AUC on a balanced subsample. Bias favours **us** on any naive reading. |
| Kepler ROC-AUC **0.9914**; pooled **0.9581** | **Islam 2026** (ExoNet) test ROC-AUC **0.9549** on a 1,139-row held-out 15%, 7,585 KOIs (2,746 CONFIRMED / 4,839 FALSE POSITIVE), 70/15/15 stratified by label | **NOT COMPARABLE** | Three independent breaks, all in our favour. (i) **Negative class**: his 4,839 FALSE POSITIVE is the *unfiltered* cumulative list — exactly the superset our cut reduces to 3,813, so §2's counterfactual applies directly and ~0.004–0.007 of our margin over him is the negative-class rule alone. (ii) **Leakage**: he deduplicates by KOI name, not Kepler ID, and the paper is **silent** on host grouping — see §5. (iii) **Domain**: the pooled 0.9581 is TESS+Kepler and cannot meet a Kepler-only number at all. |
| Pooled ROC-AUC **0.9581 ± 0.0057** | **Shallue & Vanderburg 2018** test ROC-AUC **0.988** on 1,523 held-out TCEs of 15,737 (3,600 PC / 9,596 AFP / 2,541 NTP), 80/10/10 | **NOT COMPARABLE** | Different populations on every axis: their row is a DR24 TCE, ours is a host star; their positive is the Autovetter's PC flag, ours is CONFIRMED; theirs is 23% positive, ours 44%; theirs is a single held-out test set, ours is out-of-fold with **no holdout and no protection from the eight-architecture selection in §4**. The report calls the served model "a dual-view 1-D CNN after Shallue & Vanderburg", which invites exactly this comparison; the architecture lineage is real and the performance comparison is not. |
| TESS ROC-AUC **0.9100 ± 0.0088** | **ExoMiner++** (Valizadegan 2025) TESS ROC-AUC **0.998** (best of four cells: 0.995 / 0.995 / 0.997 / 0.998), 57,162 TESS SPOC TCEs, 10-fold CV **grouped by target star** | **COMPARABLE WITH STATED CAVEAT** | The one protocol pair that genuinely lines up: both are out-of-fold k-fold cross-validation grouped by host, both TESS, both exclude PC/APC from the labelled set. The caveats are large and must be stated: their population is 93.6% negative against our 45%, 57,162 TCEs against our 2,371 rows, and they carry eleven diagnostic branches including difference imaging that our served dual-view model does not have. The gap is real and the direction is unambiguous — **we are far behind, and it is not an artefact of protocol**. |
| Pooled F1 **0.9001 ± 0.0125** at the per-fold fitted threshold; TESS F1 **0.4718** at the 1% FPR cut | **Xie et al. 2025** F1 **0.957** (Kepler) / **0.995** (TESS), both at threshold 0.5, single 9:1 split | **NOT COMPARABLE** | Our own two F1s differ by 0.43 on the same model at the same moment, which is the whole argument: F1 is a function of the threshold, and the threshold is not shared. Theirs is 0.5; ours is a per-fold F1-optimal sweep on the inner validation split. Their TESS population is 97% negative (391 PC of 13,207), where F1 on the minority class and accuracy 0.999 are near-vacuous. Their paper is **silent** on host grouping. |
| **TESS recall @1% FPR 0.3113 ± 0.0656**; Kepler 0.8129 ± 0.0511 | — | **NOT COMPARABLE — no published counterpart exists** | The metric the project actually promotes on has no match in any of the five sources. The nearest is ExoMiner's recall at *fixed precision* 99%, which is a different budget: precision fixes the cost per selected candidate, FPR fixes the cost per negative examined, and they diverge exactly where the class prior differs — which is where we differ most. **This is the most important line in the table.** The decision metric is unbenchmarkable against the literature as it stands. |

**Two comparisons survive in any form, and only one is favourable to state:**
TESS 0.9100 against ExoMiner++ 0.998 with the caveat above, which is a
comparison we lose. The other four are refusals.

## 5. Islam's deduplication is leaky with respect to host star

**Established from the paper:** it deduplicates "by KOI name (individual transit
signal) rather than by Kepler ID (host star)", and splits 70/15/15 "stratified
by label". **The paper is silent on host grouping** — it neither claims it nor
rules it out, and the reader recorded that as SILENT rather than inferring.

**Established from the archive:** 8,054 DR25 KOIs sit on 6,923 `kepid`s; 775
hosts carry more than one KOI; **1,906 KOIs (23.7%) have at least one sibling on
the same star**, up to 7 on one host.

So *if* the split is not additionally grouped — which the deduplication rule
described makes likely but does not establish — roughly 23.7% of KOIs are
sibling-bearing, and about 70% of each one's siblings land in training, putting
on the order of **16% of his test rows on a star the model trained on**. Sibling
KOIs share the star's stellar parameters, its systematics and its noise, and
ExoNet carries a stellar-parameter branch that can memorise them.

**Verdict on the question asked: our grouped 0.958 may NOT be quoted beside his
0.955.** Two reasons, and the leakage is the weaker of them. The stronger is
that 0.958 is a pooled TESS+Kepler number and 0.955 is Kepler-only; the
mission-matched figure is 0.9914, and that one is separated from his by an
unfiltered negative class worth ~0.004–0.007 plus whatever the leakage is worth.
The two numbers look 0.003 apart and are not close to measuring the same thing.

## 6. ExoMiner is the right protocol comparator, and 5,812 vs 30,609 is the reason it still does not settle anything

**Yes on protocol, for ExoMiner++ specifically.** Valizadegan 2025 is verbatim
grouped by target star — *"The target stars were divided into 10 equal subsets,
with the TCEs from one subset serving as the test set"* — and evaluated out of
fold. Valizadegan 2022 is the same design, though the reader could only
corroborate its grouping sentence through a citing follow-up paper after the
fetch truncated at *"we split TCEs by th[eir respective target stars]"*; treat
2022's grouping as strongly corroborated rather than verbatim-confirmed.

**But the protocol match does not license a number match**, for three reasons
the size ratio makes concrete:

1. **Different denominators.** Their row is a TCE; ours is a host star. 30,609
   TCEs sit on far fewer stars, and 57,162 TESS TCEs collapse to 42,875 events.
   Rate metrics computed per-TCE and per-star are different rates.
2. **Different priors.** 6.4% positive against our 45%. ROC-AUC is prior-
   insensitive in principle and PR-AUC, precision and recall are not — and it is
   precision and recall that both projects actually decide on.
3. **Different power.** This is where the ratio bites. `known-limits.md` records
   our decision metric as an **eight-to-ten row statistic** — TESS recall @1%
   FPR is cut at the 10th-highest negative score, paired sampling sd 0.0437, so
   nothing below ~0.09 is detectable. A 5.3× larger evaluation population is not
   a footnote against that; it is the difference between a measurement and an
   estimate. **Any comparison drawn to ExoMiner is a comparison between a
   stable number and a noisy one, and the noise is entirely on our side.**

## 7. Figures in `docs/report.md` that imply an unsupported comparison

The implied-comparison scan agent died with the fleet, so this list was built by
hand in the main session and should be treated as **not exhaustive**.

| line | verbatim | what a reader takes from it | supported? |
|---|---|---|---|
| 81 | "ROC-AUC is reported because it is comparable to the literature" | that our AUC can be set beside published AUCs | **No.** Of the five sources, one reports a Kepler ROC-AUC we could verify (Islam, 0.9549, unfiltered negatives and silent on grouping) and one a TESS ROC-AUC (ExoMiner++, 0.998, 93.6% negative). Neither is a like-for-like population. The sentence is the single most load-bearing overclaim in the report. |
| 21 | "a dual-view 1-D CNN after Shallue & Vanderburg" | architectural lineage — **and**, in a sentence two lines above "ROC-AUC 0.9581", a performance peer | Lineage yes, performance no. Harmless alone; not alone. |
| 23–25 | "Out of fold it reaches ROC-AUC **0.9581 ± 0.0057**" in the Summary | a headline number a CV reader compares to whatever they last read | Bare AUC in an abstract-like position invites the comparison §4 refuses. Needs the population named in the same breath. |
| 142–150 | the negative-class paragraph | that the 21% attrition is the `koi_score` rule doing ambiguity-filtering | The report is **correct** as written ("disputed or unvetted") — but it is silent on the mechanism, which is why the hypothesis in §1 was able to form. Worth stating positively. |
| 677 | "`valizadegan2022`, `valizadegan2025` — the per-diagnostic branch design §4 is *inspired by*, not a reimplementation of" | correctly scoped to architecture | **Yes.** This one is right and should be the model for the others. |

## 8. What this audit did not establish

- **The six verdicts in §4 were not adversarially refuted.** The audit design
  specified three independent refuters per verdict under distinct lenses
  (population, protocol, over-generosity). That fleet failed on a session limit
  and the verdicts were reached by hand instead. They are one reader's judgment
  on well-sourced evidence, which is weaker than this record's usual standard.
- **The implied-comparison scan of §7 was likewise done by hand** and covers
  `report.md` only — not `data_provenance.md`, `overview.md`, `README.md` or the
  console copy.
- **No ROC-AUC could be established for Valizadegan 2022** from the primary
  source; its metric definitions section truncated on every fetch.
- **Valizadegan 2022's grouping sentence is corroborated, not verbatim.**
- **Shallue & Vanderburg and Xie are both silent on host grouping.** Their
  splits are described at signal granularity, which makes star-level leakage
  likely in both, but neither paper rules it in or out and this audit does not
  assert it. Gemini's audit does assert it; see §9.
- **The counterfactual Kepler AUC is a resampling bound, not a measurement.**
  The discarded rows were never scored. Scoring them would settle it and is
  cheap — 1,026 rows through the served model.

## 9. Note on the Gemini audit of 2026-09-09

A second audit of the same question was produced on 2026-09-09 and circulated as
binding. It reaches several conclusions this one shares — Islam not comparable,
ExoMiner the best protocol match, pooled AUC never quotable against a
single-mission number. **Its central quantitative claim is false**, and because
it proposes report edits, recording the specific defects matters:

1. **"The discarded 21% consists of ambiguous or disputed false positives where
   the Robovetter score was ≥ 0.5"** — measured false. Four of 1,026. Its
   proposed Edit 1 would replace the report's currently-correct wording ("the
   rest are dropped as disputed or unvetted") with this claim, making the report
   **less** accurate than it is today. **Do not apply that edit.**
2. **"V2's Kepler ROC-AUC is estimated to drop from 0.9914 to ~0.970–0.985"** —
   no method given. The resampling bound in §2 puts it at 0.985–0.988.
3. **"ExoMiner (Valizadegan 2022) … Kepler ROC-AUC of 0.999"** — the 0.999 is
   Valizadegan **2025**'s Kepler baseline row (Table 6). No ROC-AUC for the 2022
   paper could be established at all.
4. **"Xie … severe data leakage" and "S&V 2018 used random 80/10/10 TCE splits
   (host-star leakage)"** — both papers are silent on grouping. Likely, not
   established, and an audit should not state it as fact.
5. Its line numbers for §2.2 ("lines 105-112") are wrong; the paragraph is at
   142–158.
6. It misses §3(b) subsampling and §3(c) inert grouping entirely, and it does
   not notice that **recall @1% FPR has no published counterpart** — the finding
   most consequential for what this project can claim.

## 10. Recommendations

1. Apply the §7 edits drafted for the next session. The `report.md` change with
   the highest value per word is line 81.
2. **Score the 1,026 discarded false positives** through `ca906040` and replace
   the §2 bound with a measurement. It is one bulk pass and it converts the
   audit's weakest quantitative claim into its strongest.
3. Add `islam2026` and `xie2025` to `references.bib` **only if** §4's refusals
   are stated alongside them. Citing them without the verdicts would create the
   comparison this audit exists to prevent.
4. Correct §5.1's rationale for the grouping, or make the grouping real by
   admitting sibling KOIs. The current text describes a protection that is doing
   nothing on this data.

---

# Appendix — the edits `docs/report.md` needs (deliverable 3)

For the next session to apply. Line numbers are against `docs/report.md` at
commit `5de0a58`. Each edit quotes the current text verbatim and gives finished
replacement prose — no placeholders. **Edits 1 and 4 are the two that matter;
the rest are hygiene.**

## Edit 1 — line 81, §1.2. The load-bearing overclaim. REQUIRED.

Current, lines 80–83:

```
1. **Recall @1% FPR is the decision metric.** ROC-AUC is reported because it is
   comparable to the literature, but a model can gain AUC while losing shortlist
   recall — stage 4's capacity arm did exactly that (§4, row 4) — and when the
   two disagree, recall governs.
```

Replace with:

```
1. **Recall @1% FPR is the decision metric.** ROC-AUC is reported because it is
   threshold-free and stable — a bootstrap sd of 0.0059 against the decision
   metric's 0.0437 — not because it is comparable to published numbers. It is
   not; the 2026-09-10 comparability audit found no published ROC-AUC computed
   on a population like ours, and no published counterpart at all for recall
   @1% FPR. A model can also gain AUC while losing shortlist recall — stage 4's
   capacity arm did exactly that (§4, row 4) — and when the two disagree,
   recall governs.
```

The clause "because it is comparable to the literature" is the sentence that
licenses every comparison §4 of the audit refuses. It must go.

## Edit 2 — lines 23–25, the Summary. Name the population beside the number.

Current:

```
5-fold cross-validation grouped by host star and calibrated per fold with Platt
scaling. Out of fold it reaches ROC-AUC **0.9581 ± 0.0057** and Brier
**0.0791 ± 0.0066** across five folds.
```

Replace with:

```
5-fold cross-validation grouped by host star and calibrated per fold with Platt
scaling. Out of fold, on a near-balanced 5,812-row labelled set of one signal
per host star, it reaches ROC-AUC **0.9581 ± 0.0057** and Brier
**0.0791 ± 0.0066** across five folds. That population is not the one published
transit classifiers report on, and §6.4 says what does and does not follow.
```

## Edit 3 — lines 149–150, §2.2. State the mechanism positively.

Current:

```
KOI table instead. About 79% of the bare cumulative false positives pass it; the
rest are dropped as disputed or unvetted.
```

Replace with:

```
KOI table instead. About 79% of the bare cumulative false positives pass it —
3,813 of 4,839. The 1,026 that do not are dropped almost entirely on DR25
membership rather than on the score: 893 are absent from the DR25 table and 129
are dispositioned CANDIDATE there, against **four** that are DR25 false
positives failing the 0.5 threshold. The threshold is close to inert, because
`koi_score` is near-degenerate on the class it filters — 89% of DR25 false
positives score below 0.001. Measured 2026-09-10; see the
[comparability audit](experiments/benchmark-comparability-2026-09-10.md) §2.
```

The current wording is *correct*. This edit makes it specific, so the reading
that "the cut discards the ambiguous false positives" cannot form again.

## Edit 4 — §6.4, after the Kepler–TESS gap paragraph (insert after line 503). The stated caveat. REQUIRED.

Insert as a new paragraph immediately after the paragraph ending "why
per-mission slicing is not optional.":

```
**These numbers are not benchmarks, and the audit that tried to make them into
benchmarks failed on purpose.** Of six candidate comparisons against the five
studies this report cites, one survives: TESS ROC-AUC 0.9100 against
ExoMiner++'s 0.998 [@valizadegan2025], which shares our protocol — out-of-fold
k-fold cross-validation grouped by host star — and which we lose by a wide
margin on a population 24 times larger. Every other comparison fails on
population rather than on performance. Shallue & Vanderburg [@shallue2018] and
Xie et al. score threshold-crossing events at a 23% and 3% planet prior; our row
is a host star at 44%. Islam's Kepler AUC of 0.955 rests on the *unfiltered*
cumulative false-positive list that our §2.2 rule reduces by 21%, and on a
70/15/15 split deduplicated by KOI name rather than by host — 23.7% of Kepler
KOIs have a sibling on the same star. Our Kepler 0.9914 is worth roughly
0.004–0.007 of AUC less on his negative class, before any leakage in his split
is counted. **And the metric this project actually promotes on — recall at a
fixed 1% false-positive rate — has no published counterpart in any of the five.**
ExoMiner's nearest figure fixes precision rather than false-positive rate, which
is a different budget wherever the class prior differs, and ours differs most.
The full working is in the
[comparability audit](experiments/benchmark-comparability-2026-09-10.md).
```

## Edit 5 — §5.1, lines 376–381. The grouping rationale is wrong on this data.

Current:

```
**5-fold `StratifiedGroupKFold`, grouped by host star, stratified on label.**
The grouping is the correctness-critical part: multi-planet systems and
re-observed targets contribute several rows that share a star, and a star split
across folds leaks. In the predecessor project, moving from a single 70/15/15
split to grouped k-fold cost 2–5 AUC points — that drop is the leakage being
removed, not a regression.
```

Replace with:

```
**5-fold `StratifiedGroupKFold`, grouped by host star, stratified on label.**
Leakage across folds is the correctness-critical risk: multi-planet systems and
re-observed targets contribute several rows that share a star, and a star split
across folds leaks. In the predecessor project, moving from a single 70/15/15
split to grouped k-fold cost 2–5 AUC points — that drop is the leakage being
removed, not a regression. **On the current labelled set the grouping is
belt-and-braces rather than the active protection**: `labels.parquet` carries
one signal per host TIC, so all 5,812 groups have size one and the grouped and
ungrouped partitions coincide. The protection is real but it lives in the label
builder, not the splitter. The cost is that our Kepler set holds 2,500 of 8,054
DR25 KOIs, having dropped the additional planets in every multi-planet system.
```

## Edit 6 — §2.2, after line 132 (the mission table). Say that it is a subsample.

Insert after the labelled-set table:

```
The Kepler rows are a **draw, not a census**: `catalog.py` takes exactly 1,250
positives from 2,748 cumulative CONFIRMED KOIs and exactly 1,250 negatives from
a 3,813-strong certified pool, ranked by `md5(seed:tic_id)` so that membership
survives a catalogue refresh. The hash is independent of period, depth and SNR,
so the draw carries no astrophysical bias — but every Kepler number in §6 is
measured on 45% of the available labelled Kepler population, and its sampling
variance is not reported anywhere in this report.
```

## Edit 7 — §9 and `references.bib`. Only with the verdicts attached.

`islam2026` and `xie2025` are **not currently cited anywhere in this report**
and are absent from `references.bib`. They may be added, but only carrying
their verdicts — a bare citation manufactures the comparison Edit 4 refuses.
Suggested §9 rows:

```
| `islam2026` | Kepler ROC-AUC 0.955 — cited in §6.4 to say why it is **not** comparable to ours: unfiltered negative class, deduplicated by KOI name rather than host star |
| `xie2025` | F1 0.957 / 0.995 at threshold 0.5 — cited in §6.4 to say why F1 at an unshared threshold is **not** a comparison |
```

## Do NOT apply

The 2026-09-09 Gemini audit's Edit 1 replacement text, which states that the
discarded 21% are "ambiguous or disputed false positives with `koi_score >=
0.5`". That is false — it is four KOIs, not 1,026 — and applying it would make
§2.2 less accurate than it is today. See §9 above.
