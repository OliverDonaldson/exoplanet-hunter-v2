> Moved verbatim from `docs/roadmap.md` §7 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

## 7. Change log

### 7.0 Incumbent became Champion — 2026-08-17

Ollie's wording: *"the best model will be known as the Champion model and weekly
refresh challenges the champion."* Code, tests and forward-facing docs now say
champion. **This document mostly does not, and that is the convention rather
than an omission.**

| surface | what happened |
|---|---|
| code, tests, `docs/index.md`, `model_pipeline.md`, `data_provenance.md`, `scripts/README.md` | renamed |
| `load_incumbent_summary` | still exported, an alias for `load_champion_summary` — the same object, so the two cannot diverge |
| `--incumbent-summary` | still accepted alongside `--champion-summary`, so commands recorded here keep running |
| `models/cv/incumbent-rebaselined/` | **unchanged on disk.** Renaming a directory is a data-layout change: it moves a DVC-tracked artefact and breaks every recorded command naming it. The vocabulary moved; the path did not |
| this document, sections 1–3 | **unchanged.** It is a record of what was measured and when, and rewriting the words a decision was taken in is not a rename |
| every pre-registration block | **unchanged, and never rewritten.** New vocabulary appears inside one only in square brackets, the convention the stage renumber used |
| `docs/handover-*.md` | unchanged; they are dated records of what a session believed |

Five forward-facing lines were renamed: the `W3` weakness, stage 8's ranking in
2g, 4.1a's statement of the defect, and 4.8's closing paragraph. Everything else
saying "incumbent" here is either a recorded result or a pre-registration, and
`incumbent` and `champion` refer to the same thing throughout: **whatever
`models/registry.json` currently serves.**

### 7.1 The 2026-08-14 record restructure

Recorded because a reshuffle of the evidence record is exactly the kind of
change that quietly loses a number.

**Verified mechanically:** every non-heading, non-blank line of the previous
`roadmap.md` is present in this one. Only heading text changed, plus the new
prose in *How to read this file*, 1d, 4.8 and this section.

| change | detail |
|---|---|
| numbered structure | flat `##`/`###` headings became `1`–`7` with `1a`/`3.2a` parts |
| section 3 reordered to chronological | the capacity-arm **launch** now precedes its **result** (the file had them inverted); *Observation baseline* moved ahead of the stage 8 pre-registration it motivates; stage 10.5's pre-registration moved out of the middle of the record into the forward plan at 4.1 |
| `plan-2026-08-09.md` deleted | its weakness register → 1d; its forward items → 4.2–4.7; its audits → 5.1–5.3; its totals rewritten at 4.8. The plan's own descriptions of hygiene, stage 7i and stage 8 were dropped as superseded by the recorded results in section 3 — they held estimates, not measurements. The file remains in git history |
| forward paragraphs pulled out of the record | the stage 9, 10, 11 and 12 paragraphs that sat inside the stage 8 result and the observation-baseline section moved to 4.2, 4.3, 4.5 and 4.7 |
| W1 status updated | the plan's "above every model" line no longer holds for arm P, which sits **below** its labels after propensity weighting |
| W13 status updated | the `npm audit fix` blocker is gone — `frontend/` is clean and the lockfile committed |
| totals re-costed | stage 10.5 added; the ~70 min per CV run figure replaced with the measured ~2 h |

**Amended 2026-08-15.** Stage 10.5's pre-registration and its amendment moved
from 4.1 into **3.11a/3.11b**, so they sit with the result whose reading they
fixed. 4.1 is now the *outstanding* part of that stage.

Reproduction for 3.11c is `ensemble.py` in the out-of-repo scratch directory.
It is **untracked**, and its own claim to have been written before the runs
finished cannot be verified — its mtime is three minutes after the last run
exited. The audit re-derived every number in 3.11c independently and they hold;
the floor it computed did not, and that is 3.11d.

**Not changed by either restructure:** every pre-registration block, every
measured number, and the stage-number mapping in 1c.

### 7.2 The 2026-08-15 documentation restructure

`docs/` was reorganised for a public reader: eight documents, no duplication,
no superseded copies. `HANDOVER.md` and five dated handovers were retired
(**W14** closed), along with the standalone architecture, features, deploy,
operating, pipeline-diagram and comparison documents, whose content was
consolidated into `getting-started.md`, `model_specs.md`, `model_pipeline.md`,
`overview.md` and `troubleshooting.md`. Measured findings and coverage
statistics are indexed in `data_provenance.md`, which is now the metrics ledger;
this file remains the record and the plan. Everything removed is in git history.
