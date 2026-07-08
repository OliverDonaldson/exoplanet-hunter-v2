# Orchestration — built in `feat/orchestrator`

Prefect (or Dagster — decide at branch time) DAG for the full V2 loop:

    catalogue refresh -> validation gate -> preprocess ->
    [dataset changed materially?] -> GPU-burst train -> promotion gate ->
    publish bundle + scores.parquet to R2

Design constraints fixed in advance:

* **Burst, don't idle.** The training step provisions a GPU, runs, tears
  down. Define "changed materially" precisely (e.g. ≥N new confirmed labels,
  or an explicit expansion tag) so trivial refreshes never fire a paid run.
* **Leakage guard.** A refresh that confirms a previously-unlabelled
  candidate must never move it into a split the current model is evaluated
  against — that check lives in the validation gate, not in the flow code.
* Lightweight refresh + CI stay on GitHub Actions; the orchestrator owns the
  conditional, stateful DAG.
