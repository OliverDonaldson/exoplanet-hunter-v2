# Contributing

This is a portfolio project with one maintainer, so "contributing" mostly means
*picking it up again in three weeks and not breaking it*. Everything below is a
rule the repository already enforces, or one a past session learned the hard way.

Read [`docs/PLAN.md`](docs/PLAN.md) first — it says where the project stands and
what is next. The rules a working session operates under are in
[`CLAUDE.md`](CLAUDE.md), which is the single copy; this file is the practical
half of the same thing.

## Environment

One conda environment, and it is not optional:

```bash
make env
conda activate exoplanet-hunter-v2
```

`environment.yml` pins it and `make install` does an editable install of both
`pipeline/` and `api/` into whatever environment is active.

**Activate it before running anything.** A predecessor project, V1, carries its
own code under the same package name, and running V2's entry points inside V1's
environment has silently executed the wrong trainer. Nothing in the code can
detect that for you.

Install the hooks once:

```bash
pre-commit install && pre-commit install --hook-type commit-msg
```

They run ruff, ruff-format, mypy, the usual whitespace and merge-conflict
checks, and a commit-msg hook that refuses assistant co-author trailers.

## Tests

```bash
make test          # the fast suites: pipeline + api, network and slow excluded
```

**Anything touching `train.py` or `train_branches.py` runs one process per
file.** Repeated `run_cv` calls inside a single process slow without bound —
measured at 86s, 108s, 161s, 190s on *identical* runs, with one file never
finishing in three hours. `clear_session()` was tested as the fix and falsified;
the cause is still open (W10 in [`docs/known-limits.md`](docs/known-limits.md)).
One process per file is the mitigation, and it holds.

Two conventions worth keeping:

- **Test names stay under 60 characters.** They are sentences about behaviour,
  not descriptions of implementation, but a name that runs past the terminal
  width stops being readable in a failure report.
- **Experimental arms are written outside `models/cv/`.** The weekly gate
  selects candidates from that directory, and an arm left inside it can be
  picked up as one.

## The promotion rule

**Never promote.** No `promotion_gate.py --promote`, no edit to
`models/registry.json`, no `fly deploy`, unless the maintainer asks for it in so
many words. The registry names what is *served*; changing it is a deployment,
not a code change.

A model becomes the champion only by clearing the gate on its own numbers: TESS
out-of-fold ROC-AUC strictly higher, Brier not degrading by more than 0.005, ECE
by more than 0.01, and recall @1% FPR not falling by more than the run's **own
measured floor**. The gate returns PROMOTE, REJECT, or UNRESOLVED — the last
meaning the margin is inside the floor, which is a stop-and-ask rather than a
rejection. It has correctly rejected several retrains; that is it working.

Two things follow from that and are easy to get wrong:

- **A margin smaller than its noise floor is not a result.** Every run with more
  than one member per fold measures its own floor by `2 x sd / sqrt(n)`. A floor
  belongs to the architecture and the run it was measured on — a branch-model
  floor read under dual-view numbers is a category error.
- **Pre-registration is binding.** How a result will be read is written down
  before the run finishes. A result landing outside those terms is recorded as
  falsified, never re-specified.

## Pull requests

**One PR per step of `docs/PLAN.md`**, with a written body: what changed, how it
was verified, and what happens after merge. Steps 1–8 are the delivery plan;
science stages 1–12 are a different numbering ([`docs/roadmap.md`](docs/roadmap.md)
§1c), and phases are a third.

- Progress is recorded as a row in `docs/PLAN.md` §1 and nowhere else. There are
  no handover files; a session that wants to explain itself does it in the PR body.
- No commit or PR carries an assistant co-author trailer or footer. The
  commit-msg hook refuses them.
- Push to `v2origin`. `origin` is the V1 repository.

## The record is frozen

`docs/experiments/` is appended to, never edited. A new stage gets a new dated
file and one row in its README; **a correction to an existing entry is a dated
note appended under the entry it corrects.** Nothing there is rewritten,
renumbered or summarised in place, including when it turns out to be wrong —
that is the point of it.

## Code

- **Guards raise.** No warn-and-continue, no broad `try/except`, no check that
  returns a plausible answer instead of failing. The recurring defect in this
  project's history is a number that looks right and describes the wrong
  population.
- **Comments say why, once, in one to three lines,** and point at the experiment
  file for the numbers. Comments that guard a past bug stay, and there are
  several — the clipped NLL paired with an unclipped gradient, the lightkurve
  stdout swap under threads, the TOI/PS transit-depth unit divergence. Leave them.
- **Module docstrings say what the module does and the constraint that shaped
  it**, in under 15 lines, with the measurements left in `docs/experiments/`.
- **Verify by executing.** A document claiming something works is a hypothesis.
  Run the path and check the artefact before writing "done".

## Is it fit to show?

```bash
make ready
```

That command answers the question and nothing else does — it checks the
documents a reader is entitled to find, that the report PDF is current and its
figures exist, that no doc link is dead, that the registry names a run whose
artefacts are on disk, that every step in `PLAN.md` has landed, that the tree is
clean, and that ruff, mypy and the fast suite are where they should be. It
prints **LOOKS GOOD** or **NOT YET** with the failing checks named. Criteria and
rationale: [`docs/showcase-readiness.md`](docs/showcase-readiness.md).

## Operations

The weekly refresh runs from this working tree every Saturday at 09:00 under
launchd, and it executes **whatever branch is checked out**. Leave the tree on a
branch whose pipeline code is tested. Its publish step rewrites DVC pointers;
commit those afterwards as a separate `data:` commit.
