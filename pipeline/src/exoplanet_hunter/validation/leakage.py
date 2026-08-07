"""Leakage guard for automated catalogue refreshes.

The system's most valuable evaluation asset is the since-confirmed temporal
holdout: candidates that were PC when the training catalogue closed and
flipped to confirmed afterwards. An automated refresh that silently
relabels those rows into the training set destroys that holdout AND leaks
future knowledge into training — the time-series iron rule, group edition.

The guard makes label changes explicit: a refresh may *add* new targets
freely, but any row whose label changed between catalogue versions must be
quarantined (kept out of training splits) until a deliberate dataset-version
bump moves it. `diff_label_catalogues` reports; `quarantine_tics` is what
the training path applies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_KEY = ["mission", "tic_id"]
#: Where the cumulative quarantine lives, beside the catalogues it derives from.
QUARANTINE_NAME = "quarantine.parquet"


def diff_label_catalogues(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Rows whose label changed between two catalogue versions.

    Returns columns: mission, tic_id, label_old, label_new,
    disposition_old, disposition_new. Empty frame = no flips.
    """
    cols = [*_KEY, "label", "disposition"]
    merged = old[cols].merge(new[cols], on=_KEY, how="inner", suffixes=("_old", "_new"))
    flips = merged[merged["label_old"] != merged["label_new"]]
    return flips.reset_index(drop=True)


def quarantine_tics(flips: pd.DataFrame) -> set[tuple[str, int]]:
    """(mission, tic_id) pairs that must not enter a training split.

    PC -> confirmed flips join the since-confirmed holdout; any other flip
    (label corrections, FP reversals) is suspicious enough to hold out too
    until reviewed.
    """
    return {(str(m), int(t)) for m, t in zip(flips["mission"], flips["tic_id"], strict=True)}


def assert_refresh_safe(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    max_flip_frac: float = 0.02,
) -> pd.DataFrame:
    """Gate a refresh: returns the flips frame, raises if the refresh looks wrong.

    A small trickle of flips is normal (TFOPWG reclassifies continuously); a
    large fraction flipping at once means the upstream query changed meaning
    (schema drift, unit change, wrong table) and nothing downstream should run.
    """
    flips = diff_label_catalogues(old, new)
    overlap = old.merge(new, on=_KEY, how="inner")
    if len(overlap) == 0:
        raise ValueError("refresh shares no targets with the previous catalogue — wrong table?")
    flip_frac = len(flips) / len(overlap)
    if flip_frac > max_flip_frac:
        raise ValueError(
            f"{len(flips)}/{len(overlap)} overlapping targets changed label "
            f"({flip_frac:.1%} > {max_flip_frac:.1%}) — refusing the refresh"
        )
    return flips


def record_quarantine(flips: pd.DataFrame, labels_dir: Path) -> pd.DataFrame:
    """Union this refresh's flips into the persisted quarantine, and return it.

    Cumulative rather than last-refresh-only: a row that flipped three refreshes
    ago is still not a training row, and deriving the set from the current pair
    of catalogues alone would quietly readmit it.
    """
    path = labels_dir / QUARANTINE_NAME
    previous = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=_KEY)
    combined = (
        pd.concat([previous[_KEY], flips[_KEY]], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    labels_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def load_quarantine(labels_dir: Path) -> set[tuple[str, int]]:
    """The persisted quarantine, or an empty set if no refresh has flipped a label."""
    path = labels_dir / QUARANTINE_NAME
    return quarantine_tics(pd.read_parquet(path)) if path.exists() else set()


def drop_quarantined(index: pd.DataFrame, quarantined: set[tuple[str, int]]) -> pd.DataFrame:
    """Remove quarantined targets from a training index.

    Applied at the top of cross-validation rather than per split: a flipped row
    belongs to the since-confirmed holdout, which `eval_since_confirmed.py`
    scores separately. Letting it into any fold both destroys that holdout and
    leaks a future label into training.
    """
    if not quarantined:
        return index
    keep = [
        (str(m), int(t)) not in quarantined
        for m, t in zip(index["mission"], index["tic_id"], strict=True)
    ]
    return index.loc[keep].reset_index(drop=True)
