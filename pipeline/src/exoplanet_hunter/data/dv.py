"""TESS DV product ingest — availability query + XML fetch.

SPOC's `*_dvr.xml` (~0.3 MB) carries per-sector difference images, the DV
scalars and observed/expected transit counts; `dv_xml.py` reads it.

`query_criteria` takes a list of target names, so targets are queried 40 per
round trip. Selection keeps the widest multi-sector run, which already holds
one `differenceImageResults` per sector it spans; skipped products stay in the
manifest with size and URI so a policy change needs no second query pass.

Sizing and coverage measurements are in HANDOVER.md (2026-08-01).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `_is_transient_error` is shared with the light-curve downloader on purpose: a
# MAST blip must not be cached as "this target has no DV products", and two
# copies of that marker list would drift. The 1,352 spurious "download error"
# rows are what getting this wrong costs.
from exoplanet_hunter.data.download import _is_transient_error
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: MAST product filename for a multi-sector DV report:
#: ``tess<ts>-s0042-s0046-0000000337385330-00550_dvr.xml``
_MULTI_SECTOR_RE = re.compile(r"-s(\d{4})-s(\d{4})-(\d{16})-(\d+)_dvr\.xml$")
#: ...and for a single-sector run: ``tess<ts>-s0044-0000000337385330-0215-s_dvr.xml``
#: (the trailing ``-s`` is the SPOC product suffix, present on single-sector
#: runs and absent on multi-sector ones).
_SINGLE_SECTOR_RE = re.compile(r"-s(\d{4})-(\d{16})-(\d+)(?:-\w+)?_dvr\.xml$")


@dataclass(frozen=True)
class DVProduct:
    """One `*_dvr.xml` on MAST, with the sector span parsed out of its name."""

    tic_id: int
    filename: str
    uri: str
    size_bytes: int
    sector_start: int
    sector_end: int
    run_id: int

    @property
    def is_multi_sector(self) -> bool:
        return self.sector_end > self.sector_start

    @property
    def span(self) -> int:
        return self.sector_end - self.sector_start + 1


@dataclass
class DVFetchResult:
    tic_id: int
    success: bool
    n_available: int = 0
    paths: list[Path] = field(default_factory=list)
    reason: str | None = None


def parse_dv_filename(filename: str, uri: str = "", size_bytes: int = 0) -> DVProduct | None:
    """Parse a DV report filename into a `DVProduct`, or None if it isn't one.

    The 16-digit TIC in the filename is the authoritative target id — MAST's
    `target_name` matching is looser than we want when querying in batches, so
    rows are attributed to targets from the filename, never from the query.
    """
    match = _MULTI_SECTOR_RE.search(filename)
    if match is not None:
        start, end, tic, run = match.groups()
    else:
        match = _SINGLE_SECTOR_RE.search(filename)
        if match is None:
            return None
        start, tic, run = match.groups()
        end = start
    return DVProduct(
        tic_id=int(tic),
        filename=filename,
        uri=uri,
        size_bytes=int(size_bytes),
        sector_start=int(start),
        sector_end=int(end),
        run_id=int(run),
    )


def select_products(products: list[DVProduct]) -> tuple[list[DVProduct], list[DVProduct]]:
    """Split a target's DV reports into (selected, skipped).

    Policy: the widest multi-sector run, since it carries one
    `differenceImageResults` per sector it spans; ties broken by the later run
    id, which is the more recent SPOC processing. A target with no multi-sector
    run keeps all of its single-sector ones.
    """
    if not products:
        return [], []
    multi = [p for p in products if p.is_multi_sector]
    if multi:
        best = max(multi, key=lambda p: (p.span, p.run_id))
        return [best], [p for p in products if p.filename != best.filename]
    return list(products), []


class DVArchive:
    """Resumable, manifest-tracked fetcher for TESS DV report XML.

    Mirrors `LightCurveDownloader`'s contract deliberately: a JSON manifest at
    ``cache_dir/manifest.json``, atomic tmp-replace writes, permanent failures
    cached and transient ones left to retry. The file on disk is the cache, so
    a manifest that has gone stale across machines costs a re-query, never a
    silently missing target.
    """

    def __init__(self, cache_dir: Path, batch_size: int = 40) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self._manifest_path = self.cache_dir / "manifest.json"
        self._manifest: dict[str, dict[str, Any]] = self._load_manifest()
        self._manifest_lock = threading.Lock()

    # ---------------------------------------------------------------- manifest

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text())
        except json.JSONDecodeError:
            log.warning("[dv] corrupted manifest; starting fresh")
            return {}

    def _save_manifest(self) -> None:
        tmp = self._manifest_path.with_suffix(self._manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, default=str))
        tmp.replace(self._manifest_path)

    def _target_dir(self, tic_id: int) -> Path:
        return self.cache_dir / f"tic_{tic_id}"

    def is_done(self, tic_id: int) -> bool:
        """True if this target needs no further work — fetched, or known empty.

        A cached *transient* failure never counts as done; `_record` refuses to
        persist those in the first place.
        """
        entry = self._manifest.get(str(tic_id))
        if entry is None:
            return False
        if not entry.get("success"):
            return True  # permanent: no DV products for this target
        paths = [Path(p) for p in entry.get("paths", [])]
        return bool(paths) and all(p.exists() for p in paths)

    # ------------------------------------------------------------ availability

    def query_availability(self, tic_ids: list[int]) -> dict[int, list[DVProduct]]:
        """One batched MAST round trip → every DV report for these targets.

        Targets with no DV products are present in the result with an empty
        list, so callers can tell "asked and there are none" apart from
        "never asked" — that distinction is what the presence mask is built on.
        """
        from astroquery.mast import Observations

        found: dict[int, list[DVProduct]] = {int(t): [] for t in tic_ids}
        obs = Observations.query_criteria(
            obs_collection="TESS",
            dataproduct_type="timeseries",
            target_name=[str(t) for t in tic_ids],
        )
        if len(obs) == 0:
            return found
        products = Observations.get_product_list(obs)
        unparsed: list[str] = []
        for row in products:
            filename = str(row["productFilename"])
            if not filename.endswith("_dvr.xml"):
                continue
            product = parse_dv_filename(
                filename, uri=str(row["dataURI"]), size_bytes=int(row["size"] or 0)
            )
            if product is None:
                # Loud, not silent: an unrecognised naming variant would
                # otherwise drop DV products for whole swathes of targets and
                # look exactly like "those targets have none".
                unparsed.append(filename)
                continue
            # A batch query can return neighbours; keep only what we asked for.
            if product.tic_id in found:
                found[product.tic_id].append(product)
        if unparsed:
            log.warning(
                "[dv] %d DV report names did not parse (e.g. %s) — check the filename patterns",
                len(unparsed),
                unparsed[0],
            )
        return found

    # ------------------------------------------------------------------ fetch

    def _record(
        self,
        tic_id: int,
        *,
        success: bool,
        selected: list[DVProduct],
        skipped: list[DVProduct],
        paths: list[Path],
        reason: str | None = None,
    ) -> None:
        if not success and _is_transient_error(reason):
            log.debug("[dv] TIC %d: transient failure not cached — %s", tic_id, reason)
            return
        entry: dict[str, Any] = {
            "success": success,
            "n_available": len(selected) + len(skipped),
            "paths": [str(p) for p in paths],
            # Everything we chose not to pull, with its size and URI, so a
            # change of selection policy is a download-only pass rather than
            # another 35-minute availability sweep.
            "skipped": [
                {"filename": p.filename, "uri": p.uri, "size": p.size_bytes, "span": p.span}
                for p in skipped
            ],
        }
        if reason:
            entry["reason"] = reason
        with self._manifest_lock:
            self._manifest[str(tic_id)] = entry
            self._save_manifest()

    def fetch_batch(self, tic_ids: list[int], *, force: bool = False) -> list[DVFetchResult]:
        """Query and download one batch of targets. One MAST query, N downloads."""
        from astroquery.mast import Observations

        pending = [int(t) for t in tic_ids if force or not self.is_done(int(t))]
        results: list[DVFetchResult] = []
        if not pending:
            return [DVFetchResult(tic_id=int(t), success=True) for t in tic_ids]

        try:
            available = self.query_availability(pending)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("[dv] availability query failed for %d targets: %s", len(pending), reason)
            return [DVFetchResult(tic_id=t, success=False, reason=reason) for t in pending]

        for tic_id in pending:
            products = available.get(tic_id, [])
            if not products:
                # Asked, and there are none. Permanent, and a real fact about
                # the target — this is 18.5% of the set, and the reason the
                # difference-image branch needs a presence mask.
                self._record(
                    tic_id,
                    success=False,
                    selected=[],
                    skipped=[],
                    paths=[],
                    reason="no DV products",
                )
                results.append(DVFetchResult(tic_id=tic_id, success=False, reason="no DV products"))
                continue

            selected, skipped = select_products(products)
            target_dir = self._target_dir(tic_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            paths: list[Path] = []
            failure: str | None = None
            for product in selected:
                dest = target_dir / product.filename
                if dest.exists() and not force:
                    paths.append(dest)
                    continue
                try:
                    Observations.download_file(product.uri, local_path=str(dest), cache=False)
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    log.warning("[dv] TIC %d: %s failed — %s", tic_id, product.filename, failure)
                    break
                paths.append(dest)

            if failure is not None:
                self._record(
                    tic_id,
                    success=False,
                    selected=selected,
                    skipped=skipped,
                    paths=paths,
                    reason=failure,
                )
                results.append(DVFetchResult(tic_id=tic_id, success=False, reason=failure))
                continue

            self._record(tic_id, success=True, selected=selected, skipped=skipped, paths=paths)
            results.append(
                DVFetchResult(
                    tic_id=tic_id,
                    success=True,
                    n_available=len(products),
                    paths=paths,
                )
            )
        return results
