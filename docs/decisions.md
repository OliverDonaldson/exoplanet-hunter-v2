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
| 2026-09-05 | **The branch line is closed in writing.** Phase 1 falsified the last standing explanation for stage 9's null, so no third stamp variant is commissioned and no further branch architecture work is scheduled. `ca906040` stays served | [phase 1 result](experiments/phase-1-result.md) |
| 2026-09-05 | The ai-slop-detector CI gate is removed: ruff already hard-fails on all three of its critical patterns (`B006`, `E722`, `SIM105`), so its one project-specific justification is redundant, and it has never fired | [#26](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/26) |
| 2026-09-05 | `AGENTS.md` is renamed to `docs/web-interface-guidelines.md` and stays tracked: the console README's "nine gaps closed" claim is only checkable if the standard is in the repo, and the root filename otherwise misdirects tooling that reads `AGENTS.md` as project instructions | [#27](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/27) |
| 2026-09-05 | The mypy pre-commit hook is kept. There is no 78-error baseline on the hook — it passes clean on 77 files; the 78 belong to `make type` under a different mypy version. Advisory-CI-only was rejected as strictly worse | [#28](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/28) |
| 2026-09-05 | `mlflow.db` is kept and the 912 orphaned `mlruns/` directories are deleted: the database is the only record of the served model's git SHA and 114 hyperparameters, and the orphans are unreachable by any run record or code path | [#29](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/29) |
| 2026-09-05 | The `models/cv/` run directories are kept, not archived: they are gitignored and invisible to a reviewer, only 201 MB is genuinely uncited, and moving bytes somewhere the repo does not describe would manufacture a second instance of the #31 problem | [#30](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/30) |
| 2026-09-05 | **The on-disk view-set bytes are `dvc add`ed and pushed.** The pointers name a 5,423-row 301/31 set the record calls already dead; the on-disk bytes are the 5,426-row 2001/201 set stages 8 and 10.5 trained on, and they exist in no cache. `dvc checkout` would have destroyed them | [#31](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/31) |
| 2026-09-05 | Stage 7i's limit 1 is stated as a limitation, not repaired. Changing a pre-registered instrument after it blocks a prediction is re-specification (rule 6), and the branch line is closed, so a measurable prediction 1 would answer a shelved question | [#32](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/32) |
| 2026-09-05 | W7 is accepted as unexplained **and recorded as resolved by the branch closure**: the +0.1446 is a branch-model deficit, the champion scores 0.9852 on that cell, and Kepler is 0% of the served population. Readiness exception A closes | [#33](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/33) |
| 2026-09-05 | Seed 45 is skipped and not revisited: no recorded conclusion depends on a fourth draw, and n=3 to n=4 does not retire the thin-floor limitation — roughly ten draws would, which is research, and the research is frozen | [#34](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/34) |
| 2026-09-05 | Discovery is kept as a stated deferral; Upload's two non-live modes are removed with their `why` copy folded into the endpoint grid; the Branch Evidence tab stops claiming "in progress" | [#35](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues/35) |

### The branch line, closed 2026-09-05

Recorded here because it is the decision the whole ExoMiner-inspired rebuild was
built to reach, and because "we never found out" was the only outcome that would
have been a failure.

Five branch arms were rejected as **replacements** for the dual-view champion,
every one on shortlist recall. Stage 10.5 then found the branch line's value as
a **complement**: mean-of-logits ensembling clears its bar on both arms. Stage
7i found no control-arm advantage. Stage 9 built the difference-image branch and
could not run its own primary test. Phase 1 fed the branch the one input
ExoMiner++ has and we did not, the star's own pixel position, and **the branch
still carries no signal** — 0.26x of its floor on the pre-registered statistic,
with the margin negative in every cell.

So the branch line is closed as a replacement, banked as a complement, and no
further architecture work is scheduled on it. What the project delivers is a
calibrated, explainable, reproducible vetting service, the label-bias work that
improves any model including the champion, and an evaluation apparatus that can
tell a real improvement from noise. That was named in advance as one of the two
finished states, and it is the one that happened.

**What this does not close.** W2, the model scoring the star rather than the
transit, is unresolved rather than solved — stage 9's instrument for it is
unrunnable by its own pre-registered limit (#32). Stage 7ii's attribution and
stage 11's serving parity are deferred (#40, #41), and they only matter if the
branch line is ever reopened.

### The ten open decisions, closed 2026-09-05

Taken in one session against a single test — does this help a reviewer, help the
report, or reduce risk — on the ground that the science is frozen and steps 5 to
8 are all that remain. Every claim resting on disk state, file sizes or a past
measurement was re-checked rather than repeated, and **four issue bodies turned
out to be wrong**:

- **#30** stated 862 MB across 30 run directories with 8 cited. `models/cv` is
  **699 MB** across **27** directories plus 3 `.dvc` files, **17 of 27** are
  cited by name, and only **201 MB** has never been cited. It also placed the
  stage 9 arms in `models/cv`; they are in `models/stage9/`, deliberately.
- **#31** described the drift as costing stage 10.5's bytes. The pointers were
  written on 2026-08-05 and never updated, and name a **5,423-row 301/31** view
  set the record already calls superseded and dead. The on-disk bytes are
  **5,426 rows at 2001/201**, matching the `view_shapes` that stages 8 and 10.5
  recorded, and they exist in no cache. The options were never symmetric.
- **#28** called 78 errors a baseline of the pre-commit hook. The hook passes
  clean on 77 files. The 78 are `make type` in the conda env under mypy 2.2.0
  against the hook's pinned 1.10.1, and 76 of them are `import-untyped` from
  installed-but-unstubbed third-party packages.
- **#35** scoped the question to Discovery and Upload. The console carries
  **seven** unbuilt markers, and the one that matters most is neither: the
  Branch Evidence tab is chipped "in progress" for a line closed on 2026-09-05.

Two findings were out of scope for the decisions they surfaced in and are
recorded as their own issues rather than folded in: the mypy version skew and
the inert `disallow_untyped_defs` override (from #28), and roughly **3.0 GB of
shard sets backing stage 9, Phase 1, Phase 1a and stage 8's synthetic-negative
arm that are gitignored, carry no DVC pointer and exist on one disk** (from
#30).

The register in `known-limits.md` is unchanged by this session: these are
decisions, not the work they authorise, and the DVC drift remains a real carried
limit until the `dvc add` and `dvc push` land.

## Open

None. The ten decisions that stood here were taken on 2026-09-05 and are
recorded above; issues [#26–#35](https://github.com/OliverDonaldson/exoplanet-hunter-v2/issues) are closed.

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
