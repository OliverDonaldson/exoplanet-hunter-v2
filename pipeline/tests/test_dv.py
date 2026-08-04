"""DV ingest: filename parsing, selection policy, manifest.

Filenames are real ones observed on MAST for TIC 337385330. Each test guards a
way the fetch could return plausible-but-wrong results rather than crash.
"""

from __future__ import annotations

import json

from exoplanet_hunter.data.dv import (
    DVArchive,
    DVProduct,
    parse_dv_filename,
    select_products,
)

MULTI_SHORT = "tess2021233042500-s0042-s0046-0000000337385330-00550_dvr.xml"
MULTI_LONG = "tess2018235142541-s0002-s0072-0000000337385330-00829_dvr.xml"
SINGLE = "tess2021284114741-s0044-0000000337385330-0215-s_dvr.xml"


def test_parses_multi_sector_report():
    p = parse_dv_filename(MULTI_SHORT, uri="mast:TESS/product/x", size_bytes=230936)
    assert p is not None
    assert p.tic_id == 337385330
    assert (p.sector_start, p.sector_end) == (42, 46)
    assert p.is_multi_sector and p.span == 5
    assert p.run_id == 550


def test_parses_single_sector_report():
    p = parse_dv_filename(SINGLE)
    assert p is not None
    assert p.tic_id == 337385330
    assert (p.sector_start, p.sector_end) == (44, 44)
    assert not p.is_multi_sector and p.span == 1


def test_rejects_non_dv_products():
    # The product list is mostly light curves, target pixel files and PDFs;
    # only the XML carries difference images and DV scalars.
    assert parse_dv_filename("tess2021284114741-s0044-0000000337385330-0215-s_lc.fits") is None
    assert parse_dv_filename(MULTI_SHORT.replace(".xml", ".pdf")) is None


def _product(name: str, size: int = 1000) -> DVProduct:
    p = parse_dv_filename(name, uri=f"mast:TESS/product/{name}", size_bytes=size)
    assert p is not None
    return p


def test_selection_takes_the_widest_multi_sector_run():
    # The widest run carries one differenceImageResults per sector it spans,
    # so it subsumes the narrower one — 3.8 GB over the set instead of 13.3.
    selected, skipped = select_products([_product(MULTI_SHORT), _product(MULTI_LONG)])
    assert [p.filename for p in selected] == [MULTI_LONG]
    assert [p.filename for p in skipped] == [MULTI_SHORT]


def test_selection_falls_back_to_every_single_sector_run():
    singles = [
        _product("tess2021284114741-s0044-0000000337385330-0215-s_dvr.xml"),
        _product("tess2021310001228-s0045-0000000337385330-0216-s_dvr.xml"),
    ]
    selected, skipped = select_products(singles)
    assert len(selected) == 2 and skipped == []


def test_selection_prefers_later_run_id_on_equal_span():
    a = _product("tess2021233042500-s0042-s0046-0000000337385330-00550_dvr.xml")
    b = _product("tess2021233042500-s0042-s0046-0000000337385330-00901_dvr.xml")
    selected, _ = select_products([a, b])
    assert selected[0].run_id == 901  # the more recent SPOC processing


def test_selection_of_nothing_is_empty_not_an_error():
    assert select_products([]) == ([], [])


def test_manifest_records_skipped_products_for_a_later_top_up(tmp_path):
    archive = DVArchive(tmp_path)
    fetched = tmp_path / "tic_337385330" / MULTI_LONG
    fetched.parent.mkdir(parents=True)
    fetched.write_text("<xml/>")
    archive._record(
        337385330,
        success=True,
        selected=[_product(MULTI_LONG)],
        skipped=[_product(MULTI_SHORT, size=230936)],
        paths=[fetched],
    )
    entry = json.loads((tmp_path / "manifest.json").read_text())["337385330"]
    assert entry["success"] and entry["n_available"] == 2
    # Sizes and URIs of what we passed over, so changing the policy later is a
    # download-only pass rather than another full availability sweep.
    assert entry["skipped"][0]["filename"] == MULTI_SHORT
    assert entry["skipped"][0]["size"] == 230936
    assert entry["skipped"][0]["uri"].startswith("mast:TESS/product/")


def test_no_dv_products_is_permanent_but_a_mast_blip_is_not(tmp_path):
    archive = DVArchive(tmp_path)
    archive._record(111, success=False, selected=[], skipped=[], paths=[], reason="no DV products")
    archive._record(
        222, success=False, selected=[], skipped=[], paths=[], reason="ConnectionError: Timeout"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # 18.5% of TESS targets genuinely have no DV products; that is a fact to
    # cache and to mask on, not a failure to retry forever.
    assert manifest["111"]["success"] is False
    assert archive.is_done(111)
    # A transient failure must never be pinned — the 1,352 spurious "download
    # error" rows are what caching infrastructure faults costs.
    assert "222" not in manifest
    assert not archive.is_done(222)


def test_is_done_requires_the_files_to_still_exist(tmp_path):
    archive = DVArchive(tmp_path)
    fetched = tmp_path / "tic_1" / MULTI_LONG
    fetched.parent.mkdir(parents=True)
    fetched.write_text("<xml/>")
    archive._record(1, success=True, selected=[_product(MULTI_LONG)], skipped=[], paths=[fetched])
    assert archive.is_done(1)
    fetched.unlink()
    # The file on disk is the cache: a manifest that has gone stale costs a
    # re-fetch, never a silently missing target.
    assert not DVArchive(tmp_path).is_done(1)


def test_corrupt_manifest_starts_fresh_rather_than_crashing(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    assert DVArchive(tmp_path)._manifest == {}
