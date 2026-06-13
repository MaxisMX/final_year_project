"""Tests for src.web.service.

The service calls slow, networked things (yfinance, LSTM training), so we mock
those and test OUR logic: the pipeline wiring, the per-ticker cache, error
handling for bad tickers, and the JSON-serialisable output shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.features.target import BUY
from src.web import service


def _fake_clean_frame(n: int = 200) -> pd.DataFrame:
    """A clean, labelled, feature-rich frame like _prepare_data would produce."""
    from src.models.random_forest import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=n)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    data["Close"] = 100 + rng.normal(0, 1, n).cumsum()
    data["label_5d"] = rng.integers(0, 2, n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture(autouse=True)
def _clear_cache():
    service.clear_cache()
    yield
    service.clear_cache()


def _patch_pipeline(pred_value: int = BUY):
    """Patch data prep + LSTM so get_recommendation runs without network/training."""
    clean = _fake_clean_frame()
    # Fake predictions Series aligned to the last rows.
    pred_index = clean.index[60 - 1 :]
    preds = pd.Series([pred_value] * len(pred_index), index=pred_index)
    y = pd.Series([pred_value] * len(pred_index), index=pred_index)

    return (
        patch.object(service, "_prepare_data", return_value=clean),
        patch.object(service, "train_lstm", return_value=MagicMock(name="artifacts")),
        patch.object(service, "predict_lstm", return_value=(preds, y)),
    )


def test_returns_recommendation_with_explanation() -> None:
    p1, p2, p3 = _patch_pipeline(pred_value=BUY)
    with p1, p2, p3:
        rec = service.get_recommendation("MSFT")
    assert rec.ticker == "MSFT"
    assert rec.recommendation == "BUY"
    assert len(rec.explanation.signals) == 4
    assert len(rec.price_history) > 0


def test_ticker_is_uppercased() -> None:
    p1, p2, p3 = _patch_pipeline()
    with p1, p2, p3:
        rec = service.get_recommendation("msft")
    assert rec.ticker == "MSFT"


def test_empty_ticker_raises() -> None:
    with pytest.raises(service.TickerError, match="must not be empty"):
        service.get_recommendation("   ")


def test_caches_model_per_ticker() -> None:
    """The LSTM should be trained once per ticker, then reused from cache."""
    p1, p2, p3 = _patch_pipeline()
    with p1, p2 as mock_train, p3:
        service.get_recommendation("MSFT")
        service.get_recommendation("MSFT")  # second call
        assert mock_train.call_count == 1  # trained only once


def test_cache_can_be_bypassed() -> None:
    p1, p2, p3 = _patch_pipeline()
    with p1, p2 as mock_train, p3:
        service.get_recommendation("MSFT", use_cache=False)
        service.get_recommendation("MSFT", use_cache=False)
        assert mock_train.call_count == 2


def test_insufficient_data_raises() -> None:
    short = _fake_clean_frame(n=30)  # fewer than lookback + 10
    with patch.object(service, "_prepare_data", return_value=short):
        with pytest.raises(service.TickerError, match="too little usable history"):
            service.get_recommendation("MSFT")


def test_to_dict_is_json_serialisable() -> None:
    import json

    p1, p2, p3 = _patch_pipeline()
    with p1, p2, p3:
        rec = service.get_recommendation("MSFT")
    d = rec.to_dict()
    # Should serialise without error.
    json.dumps(d)
    assert d["ticker"] == "MSFT"
    assert "signals" in d
    assert "price_history" in d


def test_bad_ticker_fetch_raises_ticker_error() -> None:
    with patch.object(
        service, "_prepare_data", side_effect=service.TickerError("bad ticker")
    ):
        with pytest.raises(service.TickerError):
            service.get_recommendation("ZZZZ")