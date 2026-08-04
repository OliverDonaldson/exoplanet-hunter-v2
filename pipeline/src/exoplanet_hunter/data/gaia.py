"""Gaia DR3 RUWE — the unresolved-binary flag the DV report does not carry.

Two hops, because the catalogues do not line up: TIC v8 gives a Gaia **DR2**
source id, and `ruwe` is a **DR3** column, so it routes through
`gaiadr3.dr2_neighbourhood`.

That table is many-to-many where DR3 resolved a blend DR2 saw as one source.
`_best_match` keeps the nearest and records how many candidates it beat —
an arbitrary row would attach a neighbour's RUWE to our target.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pandas as pd

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

TIC_CHUNK = 500
GAIA_CHUNK = 1000

_RUWE_QUERY = """
SELECT n.dr2_source_id, s.source_id AS dr3_source_id, s.ruwe,
       n.angular_distance, n.magnitude_difference
FROM gaiadr3.dr2_neighbourhood AS n
JOIN gaiadr3.gaia_source AS s ON s.source_id = n.dr3_source_id
WHERE n.dr2_source_id IN ({ids})
"""


def _chunks(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_gaia_ids(tic_ids: list[int]) -> pd.DataFrame:
    """TIC id -> Gaia DR2 source id, via the MAST TIC catalogue."""
    from astroquery.mast import Catalogs

    frames: list[pd.DataFrame] = []
    for n, chunk in enumerate(_chunks(sorted(tic_ids), TIC_CHUNK), start=1):
        table = Catalogs.query_criteria(catalog="TIC", ID=[str(t) for t in chunk])
        if len(table) == 0:
            continue
        df = table["ID", "GAIA"].to_pandas()
        df = df[df["GAIA"].notna() & (df["GAIA"].astype(str).str.strip() != "")]
        frames.append(
            pd.DataFrame(
                {
                    "tic_id": df["ID"].astype("int64"),
                    "dr2_source_id": df["GAIA"].astype("int64"),
                }
            )
        )
        log.info("[gaia] TIC lookup %d/%d chunks", n, -(-len(tic_ids) // TIC_CHUNK))
    if not frames:
        return pd.DataFrame(columns=["tic_id", "dr2_source_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("tic_id")


def _best_match(rows: pd.DataFrame) -> pd.DataFrame:
    """One DR3 row per DR2 source: the nearest, with the ambiguity recorded.

    `n_dr3_candidates > 1` means DR3 split what DR2 saw as one source. That is
    itself a blend indicator, and a consumer should be able to see it rather
    than silently inherit the nearest star's RUWE.
    """
    counts = rows.groupby("dr2_source_id").size().rename("n_dr3_candidates")
    best = rows.sort_values(["dr2_source_id", "angular_distance"]).drop_duplicates(
        "dr2_source_id", keep="first"
    )
    return best.merge(counts, on="dr2_source_id", how="left")


def fetch_ruwe(tic_ids: list[int]) -> pd.DataFrame:
    """RUWE for each TIC that has a Gaia counterpart.

    Returns tic_id, dr2_source_id, dr3_source_id, ruwe, angular_distance,
    magnitude_difference, n_dr3_candidates. TICs with no Gaia id, or no DR3
    neighbour, are simply absent — the caller masks on presence rather than
    imputing, since a missing RUWE is not a RUWE of 1.
    """
    from astroquery.gaia import Gaia

    ids = fetch_gaia_ids(tic_ids)
    if ids.empty:
        return ids.assign(dr3_source_id=None, ruwe=None)
    log.info("[gaia] %d/%d TICs have a Gaia DR2 id", len(ids), len(tic_ids))

    frames: list[pd.DataFrame] = []
    dr2 = sorted(ids["dr2_source_id"].unique())
    total = -(-len(dr2) // GAIA_CHUNK)
    for n, chunk in enumerate(_chunks(dr2, GAIA_CHUNK), start=1):
        started = time.time()
        query = _RUWE_QUERY.format(ids=",".join(str(i) for i in chunk))
        table = Gaia.launch_job_async(query).get_results()
        frames.append(table.to_pandas())
        log.info(
            "[gaia] RUWE %d/%d chunks  %d rows  %.0fs",
            n,
            total,
            len(table),
            time.time() - started,
        )
    if not frames:
        return ids.assign(dr3_source_id=None, ruwe=None)

    matched = _best_match(pd.concat(frames, ignore_index=True))
    return ids.merge(matched, on="dr2_source_id", how="inner")
