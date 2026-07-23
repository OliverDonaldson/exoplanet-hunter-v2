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
