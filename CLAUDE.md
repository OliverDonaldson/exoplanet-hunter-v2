# Exoplanet Hunter V2 — rules for every session

Read `docs/PLAN.md` first: it says where the project stands and what is next.
The record of what was measured is `docs/experiments/` (frozen), the weaknesses
`docs/known-limits.md`, the decisions `docs/decisions.md`.

1. Activate the environment first: `source /opt/anaconda3/etc/profile.d/conda.sh
   && conda activate exoplanet-hunter-v2`. The V1 environment carries V1's code
   under the same package name and has silently run the wrong trainer before.
2. Never promote. No `promotion_gate.py --promote`, no edit to
   `models/registry.json`, no `fly deploy`, unless Ollie (Oliver Donaldson, the
   maintainer) asks in the message.
3. One PR per step of `docs/PLAN.md` (steps 1–8 are the delivery plan; the
   science stages 1–12 are a different numbering, see `docs/roadmap.md` §1c). Push the stage branch and open the PR
   with a written body: what, verification, after merge. No commit or PR
   carries an assistant co-author trailer or footer; the commit-msg hook
   refuses it.
4. No handover files. Progress is a row in `docs/PLAN.md` §1; explanations go
   in the PR body.
5. The record is frozen. `docs/experiments/` is appended to, never edited; a
   correction is a dated note under the entry it corrects.
6. Pre-registration is binding. How a result will be read is written before the
   run finishes; a result outside it is reported as falsified, never
   re-specified.
7. A margin smaller than its noise floor is not a result. A floor is quoted with
   the architecture and run it was measured on; a branch-model floor is never
   read under dual-view numbers.
8. Guards raise. No warn-and-continue, no broad `try/except`, no check that
   returns a plausible answer instead of failing.
9. A harness that rebuilds inputs runs from the code version of the model it
   scores, pinned in a detached worktree, not the working tree.
10. Comments say why, once, in one to three lines, and point at the experiment
    file for the numbers. Comments that guard a past bug stay.
11. Verify by executing. A document claiming something works is a hypothesis;
    run the path and check the artefact before writing "done".
12. Tests: `make test` for the fast suites; anything touching `train*.py` runs
    one process per file. Experimental arms are written outside `models/cv/`.
13. The weekly refresh runs from this working tree every Saturday at 09:00 and
    executes whatever branch is checked out. Leave the tree on a branch whose
    pipeline code is tested, and commit the DVC pointers it rewrites as a
    separate `data:` commit.
