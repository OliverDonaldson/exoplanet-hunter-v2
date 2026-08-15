"""Recover the candidates the pipeline cannot see at all.

744 scored candidates come back `no_fits`: SPOC never produced a 2-minute light
curve for them, so they have no views, no score, and no way into the model.
They are not badly scored — they are absent. TESS also images them in the
full-frame images, and several groups publish light curves derived from those,
so this fetches one per target and puts them back in scope.

Tries TESS-SPOC first (same pipeline lineage as our 2-minute data) then QLP,
and records which author supplied each target: two detrendings mixed is a
systematic worth being able to mask on later.

Cached in `data/raw/tess/ffi/`, never beside the 2-min light curves, because FFI cadence is 200 s to
30 min against SPOC's 120 s and the two must not be confused downstream.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exoplanet_hunter.data.download import _corrupt_product_path, _is_transient_error
from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

#: Tried in order; the first author with any product wins.
AUTHORS: tuple[str, ...] = ("TESS-SPOC", "QLP")


@dataclass
class FFIResult:
    tic_id: int
    success: bool
    author: str | None = None
    n_sectors: int = 0
    n_points: int = 0
    cadence_seconds: float | None = None
    path: Path | None = None
    reason: str | None = None


class FFIDownloader:
    """Resumable fetcher: one stitched FITS per target, manifest-tracked.

    Same contract as `LightCurveDownloader` — atomic writes, permanent failures
    cached, transient ones left to retry.
    """

    def __init__(self, cache_dir: Path, authors: tuple[str, ...] = AUTHORS) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.authors = authors
        self._manifest_path = self.cache_dir / "manifest.json"
        self._manifest: dict[str, dict[str, Any]] = self._load_manifest()
        self._manifest_lock = threading.Lock()

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text())
        except json.JSONDecodeError:
            log.warning("[ffi] corrupted manifest; starting fresh")
            return {}

    def _save_manifest(self) -> None:
        tmp = self._manifest_path.with_suffix(self._manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, default=str))
        tmp.replace(self._manifest_path)

    def _target_path(self, tic_id: int) -> Path:
        return self.cache_dir / f"tic_{tic_id}.fits"

    def is_done(self, tic_id: int) -> bool:
        entry = self._manifest.get(str(tic_id))
        if entry is None:
            return False
        if not entry.get("success"):
            return True  # permanent: no FFI product from any author
        return self._target_path(tic_id).exists()

    def _record(self, tic_id: int, result: FFIResult) -> None:
        if not result.success and _is_transient_error(result.reason):
            log.debug("[ffi] TIC %d: transient failure not cached — %s", tic_id, result.reason)
            return
        entry: dict[str, Any] = {"success": result.success}
        if result.success:
            entry.update(
                {
                    "author": result.author,
                    "n_sectors": result.n_sectors,
                    "n_points": result.n_points,
                    "cadence_seconds": result.cadence_seconds,
                    "path": str(result.path),
                }
            )
        else:
            entry["reason"] = result.reason
        with self._manifest_lock:
            self._manifest[str(tic_id)] = entry
            self._save_manifest()

    def _median_cadence(self, stitched: Any) -> float | None:
        """Median spacing in seconds; distinguishes FFI from 2-minute data."""
        import numpy as np

        time = np.asarray(stitched.time.value, dtype=float)
        time = time[np.isfinite(time)]
        if time.size < 2:
            return None
        return float(np.median(np.diff(np.sort(time))) * 86400.0)

    def download_one(self, tic_id: int, force: bool = False) -> FFIResult:
        """Search each author in order, stitch the first that has products."""
        import lightkurve as lk

        target_path = self._target_path(tic_id)
        if not force and self.is_done(tic_id):
            entry = self._manifest.get(str(tic_id), {})
            return FFIResult(
                tic_id=tic_id,
                success=bool(entry.get("success")),
                author=entry.get("author"),
                n_sectors=int(entry.get("n_sectors", 0)),
                n_points=int(entry.get("n_points", 0)),
                cadence_seconds=entry.get("cadence_seconds"),
                path=target_path if entry.get("success") else None,
                reason=entry.get("reason"),
            )

        stage = self.cache_dir / ".staging" / f"tic_{tic_id}"
        last_error: str | None = None
        for author in self.authors:
            try:
                search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author=author)
            except Exception as exc:
                last_error = f"search error ({author}): {exc}"
                continue
            if len(search) == 0:
                continue

            collection = None
            for attempt in (0, 1):
                try:
                    collection = search.download_all(download_dir=str(stage))
                    break
                except Exception as exc:
                    if attempt == 1:
                        last_error = f"download error ({author}): {exc}"
                        break
                    # An interrupted download leaves a truncated FITS that
                    # poisons every retry until evicted.
                    corrupt = _corrupt_product_path(exc)
                    if corrupt is not None:
                        corrupt.unlink()
                        log.warning("[ffi] evicted corrupt cache file %s — retrying", corrupt)
            if collection is None or len(collection) == 0:
                continue

            try:
                stitched = collection.stitch()
                stitched.to_fits(target_path, overwrite=True)
            except Exception as exc:
                last_error = f"stitch/write error ({author}): {exc}"
                shutil.rmtree(stage, ignore_errors=True)
                continue

            # Per-target staging, removed on success (stage 1 deleted 71 GB
            # of stitched-and-forgotten debris from not doing this).
            shutil.rmtree(stage, ignore_errors=True)
            result = FFIResult(
                tic_id=tic_id,
                success=True,
                author=author,
                n_sectors=len(collection),
                n_points=len(stitched),
                cadence_seconds=self._median_cadence(stitched),
                path=target_path,
            )
            self._record(tic_id, result)
            return result

        shutil.rmtree(stage, ignore_errors=True)
        result = FFIResult(
            tic_id=tic_id,
            success=False,
            reason=last_error or "no FFI data from any author",
        )
        self._record(tic_id, result)
        return result
