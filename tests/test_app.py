"""
Tests for src.web.app (the Flask API).

We use Flask's test client and mock the service layer so these run fast and
offline. They verify the HTTP contract: success -> 200 + JSON, bad ticker ->
400, unexpected error -> 500, and that the index page renders.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.web import app as app_module
from src.web.service import Recommendation, TickerError
from src.explain.plain_english import explain
from src.features.target import BUY


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _fake_recommendation() -> Recommendation:
    exp = explain(BUY, close=110, sma_20=100, rsi_14=45, macd_hist=0.5, momentum_5d=0.03)
    return Recommendation(
        ticker="MSFT",
        as_of_date="2026-05-21",
        recommendation="BUY",
        explanation=exp,
        price_history=[{"date": "2026-05-20", "close": 410.5}],
    )


def test_index_page_renders(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Plainstock" in resp.data or b"plain English" in resp.data


def test_api_success_returns_json(client) -> None:
    with patch.object(app_module, "get_recommendation", return_value=_fake_recommendation()):
        resp = client.get("/api/recommend/MSFT")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ticker"] == "MSFT"
    assert data["recommendation"] == "BUY"
    assert "signals" in data
    assert "price_history" in data


def test_api_bad_ticker_returns_400(client) -> None:
    with patch.object(
        app_module, "get_recommendation", side_effect=TickerError("bad symbol")
    ):
        resp = client.get("/api/recommend/ZZZZ")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_unexpected_error_returns_500(client) -> None:
    with patch.object(
        app_module, "get_recommendation", side_effect=RuntimeError("boom")
    ):
        resp = client.get("/api/recommend/MSFT")
    assert resp.status_code == 500
    assert "error" in resp.get_json()


def test_api_error_payload_is_json(client) -> None:
    with patch.object(
        app_module, "get_recommendation", side_effect=TickerError("too little data")
    ):
        resp = client.get("/api/recommend/NEW")
    assert resp.is_json
    assert "too little data" in resp.get_json()["error"]