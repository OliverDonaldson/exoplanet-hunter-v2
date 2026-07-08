"""Tests for the candidate-catalogue endpoints against a small fixture parquet."""

import pandas as pd
import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture()
def catalogue(tmp_path, monkeypatch):
    rows = pd.DataFrame(
        [
            {
                "source": "TOI",
                "name": "TOI-101.01",
                "tic_id": 231663901,
                "disposition": "KP",
                "tess_mag": 12.4,
                "period_days": 1.43,
                "depth_ppm": 18960.7,
                "comments": "WASP-46 b",
            },
            {
                "source": "TOI",
                "name": "TOI-4328.01",
                "tic_id": 353475866,
                "disposition": "PC",
                "tess_mag": 11.9,
                "period_days": 703.0,
                "depth_ppm": 4100.0,
                "comments": "long period",
            },
            {
                "source": "CTOI",
                "name": "TIC 55650590.01",
                "tic_id": 55650590,
                "disposition": None,
                "tess_mag": 9.1,
                "period_days": 12.9,
                "depth_ppm": 890.0,
                "comments": None,
            },
        ]
    )
    path = tmp_path / "candidates.parquet"
    rows.to_parquet(path, index=False)
    monkeypatch.setenv("CATALOGUE_PATH", str(path))
    return rows


def test_candidates_missing_catalogue_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOGUE_PATH", str(tmp_path / "absent.parquet"))
    assert client.get("/candidates").status_code == 503


def test_candidates_lists_all_rows(catalogue):
    body = client.get("/candidates").json()
    assert body["total"] == 3
    assert {r["name"] for r in body["rows"]} == {"TOI-101.01", "TOI-4328.01", "TIC 55650590.01"}


def test_candidates_filters_and_sort(catalogue):
    body = client.get("/candidates", params={"disposition": "PC"}).json()
    assert [r["name"] for r in body["rows"]] == ["TOI-4328.01"]

    body = client.get("/candidates", params={"search": "wasp"}).json()
    assert [r["name"] for r in body["rows"]] == ["TOI-101.01"]

    body = client.get("/candidates", params={"disposition": "none"}).json()
    assert [r["tic_id"] for r in body["rows"]] == [55650590]

    body = client.get("/candidates", params={"sort_by": "period_days", "order": "desc"}).json()
    assert body["rows"][0]["name"] == "TOI-4328.01"

    assert client.get("/candidates", params={"sort_by": "evil; drop"}).status_code == 422


def test_candidates_pagination(catalogue):
    body = client.get("/candidates", params={"limit": 2, "offset": 2}).json()
    assert body["total"] == 3
    assert len(body["rows"]) == 1


def test_candidates_csv_export_respects_filters(catalogue):
    resp = client.get("/candidates.csv", params={"source": "CTOI"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    lines = resp.text.strip().splitlines()
    assert len(lines) == 2  # header + the one CTOI row
    assert "55650590" in lines[1]
