"""Data and model validation gates — built in `feat/validation-gates`.

Planned contents:
  * Pandera schemas for the label catalogue: column types, disposition-label
    domains, period/epoch sanity, no all-NaN folds.
  * Leakage guard: the automated catalogue refresh must never move a
    newly-confirmed label into a split it is later evaluated against.
  * Promotion gate: a training run that does not beat the current best CV
    score on the same split does not get promoted to the serving bundle.
"""
