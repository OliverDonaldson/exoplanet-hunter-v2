# Is this fit to show?

One command answers it:

```bash
make ready          # the repository
make ready-live     # the repository plus the deployed API and console
```

It prints **LOOKS GOOD** or **NOT YET**, and when it says NOT YET it names every
failing check. Nothing else in this repository is allowed to answer the
question — in particular, no document saying the project is finished counts.
That is the same rule the delivery steps run under: a step cannot be declared
done by writing about it.

## What "ready" means here

Ready to **show**, not ready to be finished. The bar is:

> A stranger can open the repository, read `docs/report.pdf`, understand what
> the model does and what it does not, and check any number in it against a
> file on disk — without asking the author anything.

That bar is deliberately about *honesty and legibility*, not about the science
being complete. The branch line is closed with a null result; the classical
baseline was never scored; TESS recall is 0.31 at a 1% false-positive rate. None
of those stops the project being worth showing. What would stop it is a reader
finding a claim they cannot check, a dead link, a figure that does not exist, or
a number on the live site the API never served.

## The checks, and why each one is there

| Check | Why it gates showing the project |
|---|---|
| documents a reader expects | README, LICENSE, CONTRIBUTING, index, PLAN, report (md **and** pdf), known-limits, decisions, the experiment index. A reader who cannot find the limitations is being sold something. |
| report PDF is current | A PDF older than its source is a document that says something the repository no longer does. |
| report figures exist | Every figure the report references resolves. A missing figure in a PDF is the most visible possible defect. |
| every doc link resolves | One dead link tells a reviewer the docs are not maintained, and they are right. |
| registry points at a real run | `models/registry.json` must name a run whose `cv_summary.json` and `predictions.parquet` are on disk. Otherwise no number in the report can be traced. |
| every delivery step landed | Reads the status table in [PLAN.md](PLAN.md). No step still "not started" or "in progress". |
| working tree is clean | Whatever is shown must be what is committed. |
| ruff clean | The lint gate CI runs. |
| mypy at or under baseline | mypy is not yet a CI gate here (issue #55, config skew), so the standard is the recorded count in `.mypy-baseline` — lower it, never raise it. |
| fast suite green | `pipeline/tests` without network or slow markers, plus `api/tests`, which includes the console-contract test. Skipped by `--quick`. |
| deployed API answers | `--live` only. A linked project that 404s is worse than no link. |
| deployed console answers | `--live` only. Same reason. |

## What it deliberately does not check

**Whether the science is good.** No check here reads a metric and judges it.
The report states TESS recall @1% FPR of 0.31 plainly, and that number is the
result, not a failure to fix before showing.

**Whether the model is state of the art.** It is not, and the report says so.

**Test coverage percentage, comment density, or any other proxy.** Step 8 of
[PLAN.md](PLAN.md) has its own exit criteria for those. They are review
standards, not publication standards, and conflating them would mean the project
is never showable.

## Before posting it publicly

Two things the script cannot check, worth a manual pass:

1. **Read the report's first page as a stranger would.** If the summary
   overclaims, fix the summary — that is the paragraph that gets quoted.
2. **Run `make ready-live`.** The Fly machine scales to zero, so the first
   request after idle takes a few seconds; a visitor arriving cold sees the same
   wait. The console's boot overlay is written to cover exactly that.
