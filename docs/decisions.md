# Decisions

Decisions that shape the work, with their reasoning, so they are not re-argued
from scratch. Taken decisions are dated and point at the record entry that
holds the reasoning. Open decisions are Ollie's; the default column says what
happens while each stays untaken. The considered-and-deferred record at the end
is moved verbatim from `roadmap.md` §6 and frozen.

## Taken

| date | decision | record |
|---|---|---|
| 2026-08-08 | Stage numbers renumbered once, to 1–12, and never again; run directories keep their old labels | `roadmap.md` §1c |
| 2026-08-13 | The console reports TESS as the gating mission; Kepler and K2 are diagnostic slices, never pooled into a headline | [3.10a](experiments/ui-scaffold-audit-2026-08-13.md) |
| 2026-08-15 | Stage 10.5's ensemble result is banked as a complement to the champion, not a replacement; nothing promotes | [3.11c–e](experiments/stage-10-5-ensemble.md) |
| 2026-08-16 | The promotion gate gains a third verdict, UNRESOLVED, and the K2 population alarm is permanently acknowledged | [4.1b](experiments/refresh-gate-calibration-4-1.md) |
| 2026-08-17 | The incumbent is renamed the Champion; the weekly refresh challenges it | [change log 7.0](experiments/change-log.md) |
| 2026-08-20 | Four phases pre-registered; the console moves first as Phase 0 | [4.2c](experiments/phases-pre-registration-4-2c.md) |
| 2026-09-04 | Freeze the science after Phase 1, finish the product, write the report | [`PLAN.md`](PLAN.md) |
| 2026-09-04 | Git history stays as it is; a commit-msg hook refuses assistant co-author trailers from now on | `scripts-dev/reject-assistant-trailer.sh` |
| 2026-09-04 | The Phase 1 arms run once, from code pinned at `d93a1a0`, and their reading closes the branch line either way | `PLAN.md` step 4 |
| 2026-09-04 | The React console (`frontend/src`) is deleted; `frontend/design-console` is the console | PR #2 |

## Open

| decision | issue | options | while untaken |
|---|---|---|---|
| The ai-slop-detector CI gate, a third-party scanner that scores code structure against a tuned threshold (`.slopconfig.yaml`, Makefile targets) | #26 | keep · remove | kept |
| `AGENTS.md` at the repo root, a generic web-interface guideline sheet for assistants | #27 | move under the ignored `.agents/` · delete · keep | kept |
| The mypy pre-commit hook with its 78-error baseline | #28 | keep · advisory CI job only | kept |
| MLflow tracking (`mlruns/`, 1.7 GB, and `mlflow.db`) | #29 | keep · drop | kept |
| 22 of 30 `models/cv/` run directories no result cites | #30 | archive to R2 or external disk · keep | kept |
| The DVC drift on the three view-set artefacts | #31 | `dvc add` the on-disk bytes · `dvc checkout` the pointers and record that stage 10.5's bytes are gone | untouched, stated in `known-limits.md` |
| Stage 7i's limit 1: the control-arm harness zeroes every DV input, which makes stage 9's primary test unrunnable; feeding real DV stamps would make it runnable at the cost of changing a pre-registered instrument | #32 | change the harness, which is re-specification · state as a limit | stated as a limit |
| W7, the narrow-span high-count Kepler cell where three architectures lose +0.1446 to the champion | #33 | investigate · accept unexplained | accepted, stated |
| Seed 45, a fourth Phase 1a draw (about 3 h) that would firm up the three-draw floors | #34 | run · skip | skipped |
| The Discovery tab and Upload's unbuilt tiles on the console | #35 | keep as stated deferrals · hide | kept as deferrals |

## Considered and deferred (frozen)

> 2026-09-05 note on 6.2 below: the served run uses the 9-dim scalar layout, not 13; 0.958 is the pooled cross-validated figure in the registry, and the TESS out-of-fold figure that gates is 0.910.

> Moved verbatim from `docs/roadmap.md` §6 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

## 6. Considered and deferred

Decisions recorded with their reasoning, so they are not re-litigated from
scratch each time they come up.

### 6.1 Transit search on raw light curves

**Status: possible today at single-target scale; deferred as a programme.**

The distinction is between *detection* (finding a periodic dip in photometry
where no ephemeris is known) and *vetting* (deciding whether a known signal is
a planet). This project does vetting. But the detection machinery is already
present and works: `search/bls.py` provides `period_grid`,
`bls_period_search` and `bls_periodogram`, and `/score/{tic_id}?force_bls=true`
ignores the catalogue ephemeris and searches blind, so detection →
classification already runs end to end on one target.

What is missing is survey scale, which is a different engineering problem:

- **iterative multi-planet search** — mask the strongest signal, re-search the
  residual, repeat;
- **false-alarm control** — a blind search over ~5,000 trial periods will find
  peaks in pure noise, so a bootstrap or analytic FA probability is required
  (this is the still-open "statistical bootstrap FA", DV §3.5, in the review
  gaps);
- **compute** — ~200,000 two-minute targets per TESS sector and millions in the
  FFIs, against a BLS capped at 5,000 trial periods over ~20,000 cadences;
- **detrending at scale** that does not absorb the signal being searched for.

**Why it is deferred.** SPOC already runs a comprehensive detection pipeline
over all 2-minute data — that is where TOIs come from — so re-detecting what it
has already detected is unlikely to add anything. More importantly, the
measured weaknesses are all on the vetting side (the 26.4% control-arm host
pass rate; probability tracking transits captured at −0.048), and stages 2–8
have a falsifiable test for fixing them. Switching
to a harder, more crowded problem mid-solve would abandon a well-posed one.

**The niche worth remembering.** BLS assumes a repeating box, so it is weakest
exactly where this model already behaves oddly: long-period and single-transit
events. 66 of the 3,919 scored candidates have a baseline covering fewer than
two periods, and 6 of the top 20 have periods over 400 days. Single- and
duo-transit detection is an area where BLS structurally underperforms and where
the community is still finding planets. If search ever enters this project,
that is the door — not a general re-run of SPOC.

### 6.2 A large language model in the pipeline

**Status: rejected for the core pipeline.**

Periodically tempting as on-device models improve (e.g. 1-bit/ternary builds
small enough to run locally). It fails the governing rule — there is no
baseline here that an LLM beats, because no task in the pipeline is
language-shaped. The core job is binary classification of a 2,001-bin global
view, a 201-bin local view and 13 scalars; a dual-view CNN does that at 0.958
AUC in milliseconds. Transit shape is a continuous-signal problem, not a
token-sequence one.

The deployment maths is independently disqualifying: the API runs on a 2 GB Fly
machine (peak 548 MB) at roughly $1–4/month scale-to-zero, and the smallest
useful local model is several GB of weights before its KV cache.

Adjacent uses that are *not* the pipeline: literature cross-referencing for a
shortlisted target is genuinely useful and belongs in a separate tool.
Natural-language verdict summaries are already produced deterministically from
the structured diagnostics. For explainability, stage 11's per-branch occlusion
contributions are quantitative and faithful to what the model computed;
narrating them with an LLM would add a layer that can be wrong about our own
model, which is the opposite of the goal.
