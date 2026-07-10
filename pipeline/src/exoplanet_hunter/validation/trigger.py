"""The GPU-burst trigger: a precise definition of "dataset changed materially".

Training costs real money on a rented GPU, so the orchestrator only fires it
when a catalogue refresh actually moves the needle. The definition, per the
V2 architecture doc's "verify before committing" note:

  * **new confirmed labels** — targets that are label=1 in the new catalogue
    and were absent (or not yet confirmed) in the old one. These are new
    ground truth: `min_new_confirmed` or more of them justify a retrain.
  * **new false positives** count the same way (`min_new_labelled` pools
    both classes) — hard negatives teach as much as confirmations.
  * **an explicit expansion run** (`force=True`) always trains — that's the
    deliberate data-scaling path, not a routine refresh.

Label *flips* on existing targets never count toward the trigger: they are
quarantined by the leakage guard (see `leakage.py`) and join the
since-confirmed holdout instead of the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_KEY = ["mission", "tic_id"]


@dataclass(frozen=True)
class RefreshDecision:
    n_new_confirmed: int
    n_new_false_pos: int
    n_new_targets: int
    n_flips: int
    should_train: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "TRAIN" if self.should_train else "SKIP"
        return f"{verdict}: " + "; ".join(self.reasons)


def evaluate_refresh(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    min_new_labelled: int = 25,
    force: bool = False,
) -> RefreshDecision:
    """Decide whether a refreshed label catalogue warrants a training run."""
    old_keys = set(zip(old["mission"], old["tic_id"], strict=True))
    added = new[
        [(m, t) not in old_keys for m, t in zip(new["mission"], new["tic_id"], strict=True)]
    ]

    n_confirmed = int((added["label"] == 1).sum())
    n_false_pos = int((added["label"] == 0).sum())
    n_labelled = n_confirmed + n_false_pos

    from exoplanet_hunter.validation.leakage import diff_label_catalogues

    n_flips = len(diff_label_catalogues(old, new))

    reasons = [
        f"{n_confirmed} new confirmed, {n_false_pos} new false positives "
        f"({len(added)} new targets total)",
        f"{n_flips} label flips quarantined to the holdout",
    ]
    if force:
        reasons.append("explicit expansion run (force=True)")
        return RefreshDecision(n_confirmed, n_false_pos, len(added), n_flips, True, reasons)
    if n_labelled >= min_new_labelled:
        reasons.append(f"{n_labelled} new labels ≥ threshold {min_new_labelled}")
        return RefreshDecision(n_confirmed, n_false_pos, len(added), n_flips, True, reasons)
    reasons.append(
        f"{n_labelled} new labels < threshold {min_new_labelled} — not worth a GPU burst"
    )
    return RefreshDecision(n_confirmed, n_false_pos, len(added), n_flips, False, reasons)
