# Plan

Where the project stands and what is left, in order. This is the only document
a session updates to record state: one row in §1 per step, nothing else. The
record of what was measured is [`experiments/`](experiments/README.md), the
weaknesses are [`known-limits.md`](known-limits.md), the decisions are
[`decisions.md`](decisions.md), and [`roadmap.md`](roadmap.md) is the index that
maps every old section number to where its text lives. There are no handover
files; a session that wants to explain itself does it in the PR body. Every
step, defect, decision and deferred item is an issue on the repository,
collected on the [project board](https://github.com/users/OliverDonaldson/projects/2) (private, like the repository); a PR
closes the issues it resolves.

**Direction, decided 2026-09-04 and now executed: freeze the science after
Phase 1, finish the product, write the report.** Phase 1 landed on 2026-09-05
and falsified the last standing explanation for the branch line's null, so the
line is [closed in writing](decisions.md) and steps 5 to 8 are all that remain. Every attempt since August to replace the
dual-view champion was rejected. The two results that cleared their floors,
stage 8's propensity weighting and stage 10.5's ensemble, improve on or
complement the champion rather than replace it, and the roadmap's own forecast
was that the branch line closes in writing with `ca906040` still served. That
is a publishable result. The remaining research is deferred with its estimates in §4.
The project is a data-science portfolio piece built on the machine-learning
workflow taught in DATA 301 at Victoria University of Wellington: formalise the
problem, collect data, preprocess, select a model, train, evaluate, report.
Every part of that exists in code; the report is the deliverable that does not
yet exist.

Three numberings coexist and are kept apart. **Stages 1–12** are the science
stages of the record (`roadmap.md` §1c and §2a). **Steps 1–8** below are the
delivery plan. **Phases 0–3** are the pre-registered order of 2026-08-20
([4.2c](experiments/phases-pre-registration-4-2c.md)): Phase 0 the console;
Phase 1 the target-position channel and momentum-dump view, run as a mechanism
test, with Phase 1a the seed sweep that measured its floors; Phase 2 the data
fix (the Kepler cap, then the TCE as unit of analysis); Phase 3 stages 10, 7ii
and 11. This plan finishes Phase 1 and defers Phases 2 and 3.

## 1. Status

| # | step | issue | PR | state | updated |
|---|---|---|---|---|---|
| 1 | Land the Phase 1 build | #3 | [#1](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/1) | merged | 2026-09-05 |
| 2 | Repo hygiene: React console removed, README, commit-msg hook | #4 | [#2](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/2), re-targeted as [#45](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/45) | merged to main via [#51](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/51) | 2026-09-05 |
| 3 | Docs restructure: this file, `experiments/`, `known-limits.md`, `decisions.md` | #5 | [#46](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/46) | merged to main via [#51](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/51) | 2026-09-05 |
| 4 | Close the science: Phase 1 arms read, branch line closed, API floor fixed | #6 | [#48](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/48), [#49](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/49) | merged; both arms gated UNRESOLVED, Phase 1 falsified, branch line closed in writing, `registry.json` untouched | 2026-09-05 |
| 5 | Console tells the truth: wrong before empty | #7 | [#47](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/47) | merged | 2026-09-05 |
| 6 | Console: empty to filled | #8 | [#52](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/52), [#53](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/53), [#57](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/57), [#58](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/58), [#59](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/59), [#60](https://github.com/OliverDonaldson/exoplanet-hunter-v2/pull/60) | merged; API redeployed | 2026-09-06 |
| 7 | The report | #9 | this branch | in progress | 2026-09-07 |
| 8 | Peer-review pass | #10 | — | not started |  |

**Is it fit to show?** `make ready` answers that, and nothing else does. It
checks the documents a reader is entitled to find, that the report PDF is
current and its figures exist, that no doc link is dead, that the registry names
a run whose artefacts are on disk, that every step above has landed, that the
tree is clean, and that ruff, mypy and the fast suite are where they should be.
It prints **LOOKS GOOD** or **NOT YET** with the failing checks named. Criteria
and rationale: [showcase-readiness.md](showcase-readiness.md).

Served: `ca906040`, unchanged since 2026-07-19. `models/registry.json` untouched.

## 2. The steps

One PR per step, each with an exit criterion that can be checked, so a step
cannot be declared done by writing a document. Hours are attended time unless
marked compute.

**1 · Land the Phase 1 build · ≈1 h.** The Phase 1 inputs (4-channel
`difference_view`, `momentum_dump_view`), the promotion log, the weekly-refresh
repair, the `/runs` test and the amber UNRESOLVED chip, as one branch.
*Exit:* fast suite green; `registry.json` byte-identical; CI green.

**2 · Repo hygiene · 4–6 h.** Delete the React console and every reference to
it; CI builds the shipping console; README rewritten to what is served; merged
branches and worktrees pruned; a commit-msg hook that refuses assistant
trailers.
*Exit:* a stranger can clone, read the README, and know what is served and how
to reproduce its numbers; no file describes a console that does not ship.

**3 · Docs restructure · 4–6 h.** Split `roadmap.md` without rewriting its
content: the record to `experiments/`, verbatim and frozen; the weakness
register to `known-limits.md`; considered-and-deferred to `decisions.md`; the
forward plan to this file. `roadmap.md` keeps every section number as a
pointer. `CLAUDE.md` carries the session rules.
*Exit:* `docs/index.md` lists every document; no link points at a missing file;
this file is under 300 lines.

**4 · Close the science · 2–3 h + ≈6 h compute.** An arm is one full
cross-validated training run of the branch model; arms come in pairs that
differ in one thing. C″ has the difference-image branch dropped at the model,
D″ keeps it; the double primes mark the Phase 1 rebuild of stage 9's arms C and
D on the Phase 1 shard set, code pinned at `d93a1a0`. Seed 44's control arm
(the harness that scores hosts with no injected transit) runs alongside. Gate
both with `--champion-summary
models/cv/champion-rebaselined-today/cv_summary.json`: the champion's own
summary predates per-mission slicing, and that file is the champion re-scored
out of fold on the current labelled set (2,367 TESS and 2,238 Kepler rows,
written 2026-08-17; `incumbent-rebaselined/` is the same population under the
old name, and neither directory name is a date). The hypothesis under test,
finding 2 of the ExoMiner++ readout in
[stage 9](experiments/stage-09-difference-image.md), is that stage 9's branch
carried no signal because its stamps were fed without the star's own pixel
position, so the network had no reference frame for a centroid shift. Read the
arms exactly against [4.2c](experiments/phases-pre-registration-4-2c.md): if
TESS recall @1% FPR on `dv_usable` rows does not move beyond the Phase 1a floor
of 0.0432, that explanation is falsified and no third stamp variant is
commissioned. Write the reading into `experiments/`, and the closing
decision on the branch line into `decisions.md`: the stage 10.5 ensemble
result stands as a complement finding; nothing promotes. Fix the floor the API
serves: read it from the champion summary's variance block when one exists,
otherwise report it as not measured for a single-member run alongside the 4.1a
dual-view figure and its source. Never a branch floor under dual-view numbers.
*Exit:* both arms carry a `promotion_log.json`; the pre-registered outcome table
is filled in; the branch line has a written status; `registry.json` untouched.

**5 · Console, wrong before empty · 4–6 h.** Mission copy and counts (vetting,
weekly, 20–60 s, real catalogue totals); insolation and habitable zone from the
catalogue row instead of a solar-mass assumption; the Kepler source filter,
APC and FA chips, the real total; `evaluation` reported rather than hardcoded.
*Exit:* no literal on the Model page that the API did not serve.

**6 · Console, empty to filled · 4–6 h.** `n_positive` on `MissionMetrics`; a
per-threshold ROC and the confusion matrix at 1% FPR computed in
`routes/model.py` from `predictions.parquet`; PR-AUC, F1 and threshold
rendered; periodogram requested; baseline on the row; fold and MC spread,
centroid components and BLS power surfaced; the "coming" tiles removed from
Upload; Discovery kept as an explicit deferral. A contract test that checks the
console client's field list against the Pydantic models.
*Exit:* every "not measured" on the Vetting page names a field that genuinely
does not exist; any screenshot carries `model_version`.

**7 · The report · 10–15 h.** `docs/report.md` rendered to PDF, sectioned on the
DATA 301 workflow: problem, data, preprocessing, model selection, training,
evaluation, results and limitations, reproducibility. Model selection is the
condensed record as one table: random forest, dual-view, the rejected branch
arms, stage 8's propensity result, stage 10.5's ensemble, stage 9 and Phase 1,
each with metric, margin, floor and verdict. Evaluation uses the served model:
ROC with the operating point marked at unit aspect ratio, confusion matrix,
precision, recall, F1, PR-AUC, calibration, and why recall @1% FPR and
calibration are the right metrics for this cost structure. Figures regenerated
from `ca906040` with `make_performance_figures.py`. The About page is written
from the report.
*Exit:* a DATA 301 marker could grade the project from the PDF alone; every
number traces to a file in `models/` or `experiments/`.

**8 · Peer-review pass · 4–6 h.** Module docstrings cut to what the module does
and the constraint that shaped it, with a pointer to the experiment file for
the numbers; every comment that guards a past bug stays. Test names under 60
characters. `docs/index.md` labels every module serving or experimental.
`CONTRIBUTING.md`: environment, tests one process per file, the promotion rule,
the PR-per-stage rule.
*Exit:* comment-plus-docstring lines under 25% of the library; no module
docstring over 15 lines; ruff, the mypy baseline and the fast suite unchanged.

## 3. Rules for every session

The rules every session works under are in [`../CLAUDE.md`](../CLAUDE.md),
the single copy; they are not repeated here so the two cannot drift.

## 4. Deferred, in writing

Each goes into the report's future-work section with its estimate. None runs
before the report exists. If any ever runs, the order is the dependency order
in [execution-order-2026-08-09](experiments/execution-order-2026-08-09.md),
with the floors re-measured after any data change.

| item | why it waits | estimate |
|---|---|---:|
| Phase 2 Tier 1: lift the Kepler cap to full DR25 (#36) | re-bases every floor and recorded margin | ≈4 h |
| Phase 2 Tier 2: the TCE as unit of analysis (#37) | the only intervention aimed at W1 at source; a research programme | ≈40 h |
| Phase 1a re-run after Phase 2 (#38) | required by Phase 2's own limit | ≈7 h compute |
| Stage 10: Optuna re-tune (#39) | tunes to a distribution Phase 2 would change | 12–15 h |
| Stage 7ii: branch attribution (#40) | describes a finished branch set; the first sweep returned four nulls | ≈8 h compute |
| Stage 11: serve the branch model, per-branch occlusion through `/score` (#41) | only matters if the branch line stays open | 12–18 h |
| Discovery: a job queue for blind BLS search (#42) | detection is out of scope by [decision 6.1](decisions.md) | unestimated |
| Light-curve upload, coordinate lookup, training-history persistence, a versioned container with a DOI, a catalogue with a DOI (#43) | product and publishing work with no bearing on the workflow the report demonstrates | 1–8 h each |

The forward plan as it stood on 2026-08-16, with its costs, is kept verbatim in
[forward-plan-2026-08-16](experiments/forward-plan-2026-08-16.md).

## 5. Phase 2 — the measurement foundation

Opened 2026-09-07 by a bootstrap of the decision metric. It is the answer to why
so much of the model-selection record reads "within floor", and it changes what
the next experiment should be.

**The finding.** TESS recall @1% FPR is cut at the 10th-highest negative score
(the 8th on `dv_usable`). A paired bootstrap of the Phase 1 contrast gives the
difference a sampling sd of **0.0437**; twice that is 0.087, the same order as
the seed floors the record reads against. Three members of one arm, identical
data and architecture, span 0.1418 to 0.2221. **Any architecture effect below
about 0.09 on this metric is undetectable at this sample size, whatever is
trained.** Stage 4's rejections are real at up to 4.8x their floor; most of what
follows them is underpowered rather than negative, and the gate returning
UNRESOLVED was the gate saying exactly that. Carried in
[known-limits.md](known-limits.md); the reasoning is [report.md](report.md) §6.4.

**What follows from it.** Build the instrument before running more models. The
order below is deliberate: each item is a prerequisite for the one after, and
nothing here trains a competitive model until item 6.

| # | item | exit criterion | cost |
|---|---|---|---:|
| P2.1 | **Power analysis, written up as an experiment file.** What effect size is detectable at what n and member count, for each candidate metric. Decide the promotion metric. | An experiment file naming the metric the gate will use and the minimum detectable effect at the current n | ≈4 h |
| P2.2 | **The gate reads a confidence interval.** Paired bootstrap on the contrast, not a seed floor alone. | `promotion_gate.py` emits a CI on the contrast; the three verdicts are decided against it; existing recorded verdicts re-checked and any that change are noted, not rewritten | ≈6 h |
| P2.3 | **Score the random forest.** `handcrafted.py::extract_features`, 14 features, the same folds and the same protocol. | [report.md](report.md) §4 row 1 carries a number instead of a dash | ≈2 h |
| P2.4 | **Scale injection-recovery.** 40 hosts to a few hundred, with the null-injection floor reported beside every completeness figure. | A completeness curve with se < 0.02 per S/N bin, and the S/N = 0 floor printed on the same axes | ≈3 h + compute |
| P2.5 | **Preprocessing and leakage audit against the new instrument.** | Every preprocessing step verified on cases with a known answer; a leakage probe that would fail if host grouping broke | ≈6 h |
| P2.6 | **One full clean run.** All 13 aux dims, the current shard set, at least 5 members per fold, both architectures, one protocol, one report. | Both architectures compared on P2.1's metric with a margin read against P2.2's interval; the answer means something either way | ≈2 h + 12–20 h compute |

**The metric recommendation, for P2.1 to accept or reject.** Partial AUC over
FPR in [0, 0.1] — every row in the follow-up-relevant region rather than one
crossing point — plus injection completeness above the null floor. Recall @1%
FPR stays reported, demoted from gating. ROC-AUC's bootstrap sd on the same rows
is 0.0059, seven times more stable than recall @1% FPR's 0.0410; the point is not
to gate on AUC but to stop gating on the least stable statistic available.

**Two re-scopings that fall out of this.** The weekly refresh is drift detection
and calibration monitoring, not a promotion path — it has never been able to
promote and should say so. And the eventual discovery goal is served by the same
work: injection-recovery is completeness as a function of depth and period, which
is what a search has to report.

## 6. Operations

- The weekly refresh runs from this working tree every Saturday at 09:00
  (launchd). It runs whatever branch is checked out, so the tree stays on a
  branch whose pipeline code is tested. Its publish step rewrites DVC pointers;
  commit them afterwards as a separate `data:` commit.
- Long runs are launched under `screen` with `caffeinate` from drivers in the
  ignored `.phase1-scratch/`, on mains power only, one process per CV run.
  Progress files: `gpu-progress.txt`, `cpu-progress.txt`.
- Experimental arms are written outside `models/cv/` so the weekly gate cannot
  select them as candidates.
