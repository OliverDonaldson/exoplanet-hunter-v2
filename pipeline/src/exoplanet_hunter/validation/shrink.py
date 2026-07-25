"""Shrink guard for automated catalogue refreshes.

The leakage guard only compares labels on targets present in *both* catalogue
versions, so it is blind to rows that simply vanish. On 2026-07-25 a capped
rebuild rewrote the data-of-record from 5,686 rows (2,656 TESS / 2,500 Kepler /
530 K2) down to 1,000 TESS-only rows and every gate reported PASS: the schema
was satisfied, the views on disk were stale but valid, and the leakage guard
saw no flips among the 1,000 survivors.

Legitimate shrink happens — the Step 2b DR25 certification rebuild retired
~21% of bare Kepler false positives on purpose — so this reports rather than
decides, and the caller says whether the reduction was intended.
"""

from __future__ import annotations

import pandas as pd


def check_catalogue_shrink(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    max_shrink_frac: float = 0.10,
) -> list[str]:
    """Row-count and mission-coverage checks across two catalogue versions.

    Returns problems (empty = pass). A mission losing every row is always a
    problem regardless of the row-count threshold: dropping one of three
    missions can stay inside the fraction while destroying a whole slice.
    """
    problems: list[str] = []
    if len(old) == 0:
        return problems

    shrink_frac = (len(old) - len(new)) / len(old)
    if shrink_frac > max_shrink_frac:
        problems.append(
            f"row count fell {len(old)} -> {len(new)} "
            f"({shrink_frac:.1%} of the previous catalogue, limit {max_shrink_frac:.1%})"
        )

    old_counts = old["mission"].value_counts()
    surviving = set(new["mission"].unique())
    gone = [str(m) for m in old_counts.index if m not in surviving]
    if gone:
        detail = ", ".join(f"{m} ({int(old_counts[m])} rows)" for m in gone)
        problems.append(f"missions absent from the new catalogue: {detail}")

    return problems
