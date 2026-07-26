"""Regression tests for interrupted-download self-healing (2026-07-09 bug).

A truncated sector file in lightkurve's cache made every score of TIC
272086938 fail, and the manifest pinned the failure as permanent so
retries short-circuited. The fix has two halves, both covered here.
"""

from exoplanet_hunter.data.download import _corrupt_product_path, _is_transient_error


def test_corrupt_product_path_extracted_and_existing(tmp_path):
    fits = tmp_path / "tess2019032160000-s0008-x_lc.fits"
    fits.write_bytes(b"truncated")
    exc = Exception(
        f"Error in reading Data product {fits} of type TessLightCurve . "
        "This file may be corrupt due to an interrupted download. "
        "Please remove it from your disk and try again."
    )
    assert _corrupt_product_path(exc) == fits


def test_corrupt_product_path_none_when_missing_or_unrelated(tmp_path):
    gone = tmp_path / "nope.fits"
    exc = Exception(f"Error in reading Data product {gone} of type TessLightCurve .")
    assert _corrupt_product_path(exc) is None  # already deleted -> nothing to evict
    assert _corrupt_product_path(Exception("no pipeline data")) is None


def test_interrupted_download_symptoms_are_transient():
    assert _is_transient_error("download error: ... This file may be corrupt due to ...")
    assert _is_transient_error("download error: I/O operation on closed file.")
    assert not _is_transient_error("no pipeline data")  # genuinely permanent


def test_existing_file_is_a_cache_hit_despite_stale_manifest(tmp_path):
    # Manifests record absolute paths, which go stale across machines; a
    # stale miss used to re-download and rewrite a FITS another request may
    # have memory-mapped (SIGBUS on the serving box).
    import json

    from exoplanet_hunter.data.download import LightCurveDownloader

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "tic_295413003.fits").write_bytes(b"cached bytes")
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "TESS:295413003": {
                    "success": True,
                    "path": "/machine/that/no/longer/exists/tic_295413003.fits",
                    "n_sectors": 3,
                    "n_points": 18881,
                }
            }
        )
    )

    result = LightCurveDownloader(raw, author="SPOC", cadence=120).download_one(295413003)
    assert result.success
    assert result.path == raw / "tic_295413003.fits"
    assert result.n_sectors == 3  # metadata still comes from the manifest


def test_manifest_survives_concurrent_downloads(tmp_path, monkeypatch):
    """Parallel download_one on distinct targets must not corrupt the shared
    manifest: without the lock, a mutation during another thread's json.dumps
    iteration raises "dictionary changed size during iteration", and a
    non-atomic write could leave a torn file."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from exoplanet_hunter.data.download import LightCurveDownloader

    class _FakeLC:
        def __len__(self):
            return 100

        def to_fits(self, path, overwrite=True):
            Path(path).write_bytes(b"fake fits")

    class _FakeCollection:
        def __len__(self):
            return 2

        def stitch(self):
            return _FakeLC()

    # Fake the fetch so download_one runs its real success path — stitch,
    # to_fits, and the lock-guarded manifest write — without any network.
    monkeypatch.setattr(
        LightCurveDownloader,
        "_fetch_kepler_via_direct_archive",
        lambda self, target_id, dl_dir: _FakeCollection(),
    )

    dl = LightCurveDownloader(tmp_path, author="Kepler", cadence=None)
    kics = list(range(1000, 1128))  # 128 distinct targets

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda k: dl.download_one(k, mission="Kepler"), kics))

    assert all(r.success for r in results)
    # Every write landed, the on-disk manifest is valid JSON, and the atomic
    # tmp-replace left no debris.
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert len(on_disk) == len(kics)
    assert all(f"Kepler:{k}" in on_disk for k in kics)
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_download_many_parallel_dedupes_targets(tmp_path, monkeypatch):
    """workers>1 collapses duplicate (mission, target_id) pairs so the same
    FITS is never fetched twice, and preserves input order in the results."""
    from pathlib import Path

    from exoplanet_hunter.data.download import LightCurveDownloader

    fetched: list[int] = []

    def fake_fetch(self, target_id, dl_dir):
        fetched.append(target_id)

        class _LC:
            def __len__(self):
                return 100

            def to_fits(self, path, overwrite=True):
                Path(path).write_bytes(b"x")

        class _Coll:
            def __len__(self):
                return 1

            def stitch(self):
                return _LC()

        return _Coll()

    monkeypatch.setattr(LightCurveDownloader, "_fetch_kepler_via_direct_archive", fake_fetch)

    dl = LightCurveDownloader(tmp_path, author="Kepler", cadence=None)
    ids = [10, 11, 10, 12, 11]  # 10 and 11 repeat
    results = dl.download_many(ids, missions=["Kepler"] * len(ids), workers=4)

    assert sorted(fetched) == [10, 11, 12]  # each distinct target fetched once
    assert [r.target_id for r in results] == [10, 11, 12]  # first-seen order


def test_k2_mission_config_uses_epic_prefix_and_shared_cache(tmp_path):
    """K2 flows through the lightkurve search path (EPIC-indexed, author "K2"),
    with FITS in the default cache under an epic_ prefix — not the Kepler
    direct-archive branch or the Kepler cache dir."""
    from exoplanet_hunter.data.download import LightCurveDownloader

    dl = LightCurveDownloader(tmp_path, kepler_cache_dir=tmp_path / "kepler")
    cfg = dl._MISSION_CFG["K2"]
    assert (cfg["search"], cfg["author"], cfg["mission"]) == ("EPIC", "K2", "K2")
    assert dl._target_path(211390903, mission="K2") == tmp_path / "epic_211390903.fits"


def test_parallel_download_leaves_stdout_usable(tmp_path, monkeypatch):
    """lightkurve's @suppress_stdout swaps a process-global under threads.

    Two workers interleave its save/restore, one closes the devnull the other
    saved as "original", and sys.stdout is left closed for the rest of the
    process — which turned good downloads into "download error" and aborted
    the run at tqdm's flush.
    """
    import io
    import os
    import sys
    import threading
    from functools import wraps

    from exoplanet_hunter.data.download import _lightkurve_stdout_swap_disabled

    # lightkurve/utils.py:558 verbatim — the decorator under test.
    def suppress_stdout(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with open(os.devnull, "w") as devnull:
                old_out = sys.stdout
                sys.stdout = devnull
                try:
                    return f(*args, **kwargs)
                finally:
                    sys.stdout = old_out

        return wrapper

    state: dict[str, threading.Event] = {}

    class FakeSearchResult:
        def download_all(self, idx):  # decorated below, like SearchResult
            # The bug needs a specific order, and *both* halves of it matter:
            # thread 0 must swap before thread 1 saves (so 1 saves 0's devnull),
            # and thread 0 must fully exit — restoring, and closing that devnull
            # — before thread 1 restores it. Sequencing only the second half
            # leaves the first to chance, which is a coin flip on CI.
            if idx == 0:
                state["first_swapped"].set()
                state["second_swapped"].wait(timeout=5)
            else:
                state["second_swapped"].set()
                state["first_done"].wait(timeout=5)
            return "lc"

    FakeSearchResult.download_all = suppress_stdout(FakeSearchResult.download_all)

    fake_module = type(sys)("lightkurve.search")
    fake_module.SearchResult = FakeSearchResult
    monkeypatch.setitem(sys.modules, "lightkurve.search", fake_module)

    # Never race pytest's own stdout: substitute a sentinel monkeypatch restores.
    sentinel = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sentinel)

    def race() -> None:
        for key in ("first_swapped", "second_swapped", "first_done"):
            state[key] = threading.Event()
        first = threading.Thread(target=FakeSearchResult().download_all, args=(0,))
        second = threading.Thread(target=FakeSearchResult().download_all, args=(1,))
        first.start()
        state["first_swapped"].wait(timeout=5)  # 0 has swapped before 1 saves
        second.start()
        first.join(timeout=5)  # 0 restored the sentinel and closed its devnull
        state["first_done"].set()
        second.join(timeout=5)  # 1 restores that closed devnull

    # Unguarded: the second thread restores the devnull the first just closed.
    race()
    assert sys.stdout is not sentinel
    assert sys.stdout.closed, "expected the unguarded race to close sys.stdout"
    sys.stdout = sentinel

    # Guarded: suppression is lifted, so nothing swaps the global at all.
    with _lightkurve_stdout_swap_disabled():
        race()
        assert sys.stdout is sentinel
        assert not sys.stdout.closed

    assert sys.stdout is sentinel
    assert not sys.stdout.closed
    # The decorator is put back afterwards.
    assert hasattr(FakeSearchResult.download_all, "__wrapped__")


def test_staging_dir_removed_after_successful_stitch(tmp_path, monkeypatch):
    """Staged per-sector products are debris once stitched (71 GB of it, once).

    Staging must be per-target so parallel workers can't delete each other's
    in-flight files, and each target's dir must be gone after its stitch.
    """
    from pathlib import Path

    from exoplanet_hunter.data.download import LightCurveDownloader

    staged: list[Path] = []

    def fake_fetch(self, target_id, dl_dir):
        # Simulate lightkurve/astroquery staging a per-sector product.
        dl_dir.mkdir(parents=True, exist_ok=True)
        (dl_dir / f"kplr{target_id}-q01_llc.fits").write_bytes(b"sector bytes")
        staged.append(dl_dir)

        class _LC:
            def __len__(self):
                return 100

            def to_fits(self, path, overwrite=True):
                Path(path).write_bytes(b"stitched")

        class _Coll:
            def __len__(self):
                return 1

            def stitch(self):
                return _LC()

        return _Coll()

    monkeypatch.setattr(LightCurveDownloader, "_fetch_kepler_via_direct_archive", fake_fetch)

    dl = LightCurveDownloader(tmp_path, author="Kepler", cadence=None)
    results = dl.download_many([111, 222], missions=["Kepler"] * 2, workers=2)

    assert all(r.success for r in results)
    assert [p.name for p in sorted(staged)] == ["kic_111", "kic_222"]  # per-target dirs
    for p in staged:
        assert not p.exists(), f"staging dir {p} should be removed after stitch"
    assert (tmp_path / ".lightkurve").exists()  # the shared root is not touched
    assert (tmp_path / "kic_111.fits").read_bytes() == b"stitched"
