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
