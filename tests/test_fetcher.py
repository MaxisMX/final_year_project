"""Tests for src.data.fetcher.

We mock yfinance.download so these tests run offline and deterministically.
They verify OUR logic — column normalisation, caching, gap detection, error
handling — not Yahoo's servers. A separate manual run (python -m src.data.fetcher)
is how you confirm the live API actually works.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.data import fetcher


def _fake_yf_frame(n_days: int = 10) -> pd.DataFrame:
    """Build a DataFrame shaped exactly like modern yfinance single-ticker output.

    That means MultiIndex columns like ('Close', 'MSFT') and an 'Adj Close'
    column that our normaliser should drop.
    """
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["MSFT"]])
    data = {}
    for field in fields:
        if field == "Volume":
            data[(field, "MSFT")] = range(1_000_000, 1_000_000 + n_days)
        else:
            data[(field, "MSFT")] = [100.0 + i for i in range(n_days)]
    df = pd.DataFrame(data, index=dates, columns=columns)
    return df


def test_normalise_flattens_multiindex_and_drops_adj_close() -> None:
    raw = _fake_yf_frame()
    out = fetcher._normalise_columns(raw, "MSFT")
    assert out.columns.tolist() == fetcher.OHLCV_COLUMNS
    assert "Adj Close" not in out.columns
    assert out.index.name == "Date"


def test_normalise_raises_on_missing_columns() -> None:
    bad = pd.DataFrame({"Open": [1, 2], "Close": [1, 2]})
    with pytest.raises(ValueError, match="missing expected columns"):
        fetcher._normalise_columns(bad, "MSFT")


def test_count_missing_trading_days_on_clean_range() -> None:
    # Consecutive business days => no weekday gaps.
    df = _fake_yf_frame(5)
    df = fetcher._normalise_columns(df, "MSFT")
    assert fetcher.count_missing_trading_days(df) == 0


def test_count_missing_trading_days_detects_gap() -> None:
    df = _fake_yf_frame(10)
    df = fetcher._normalise_columns(df, "MSFT")
    # Drop two middle rows to simulate missing trading days.
    df_with_gap = df.drop(df.index[[3, 4]])
    assert fetcher.count_missing_trading_days(df_with_gap) == 2


def test_count_missing_trading_days_empty() -> None:
    assert fetcher.count_missing_trading_days(pd.DataFrame()) == 0


def test_fetch_writes_and_reads_cache(tmp_path) -> None:
    """First call downloads (mocked); second call must read cache, not re-download."""
    with patch("src.data.fetcher.yf.download", return_value=_fake_yf_frame()) as mock_dl:
        first = fetcher.fetch_ohlcv("MSFT", cache_dir=tmp_path)
        assert mock_dl.call_count == 1
        assert first.columns.tolist() == fetcher.OHLCV_COLUMNS

        # Cache file should now exist.
        assert (tmp_path / "MSFT_ohlcv.parquet").exists()

        # Second call: download must NOT be called again.
        second = fetcher.fetch_ohlcv("MSFT", cache_dir=tmp_path)
        assert mock_dl.call_count == 1
        pd.testing.assert_frame_equal(first, second)


def test_force_refresh_bypasses_cache(tmp_path) -> None:
    with patch("src.data.fetcher.yf.download", return_value=_fake_yf_frame()) as mock_dl:
        fetcher.fetch_ohlcv("MSFT", cache_dir=tmp_path)
        fetcher.fetch_ohlcv("MSFT", cache_dir=tmp_path, force_refresh=True)
        assert mock_dl.call_count == 2


def test_fetch_raises_on_empty_download(tmp_path) -> None:
    with patch("src.data.fetcher.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data returned"):
            fetcher.fetch_ohlcv("FAKE", cache_dir=tmp_path)


def test_ticker_is_case_insensitive(tmp_path) -> None:
    with patch("src.data.fetcher.yf.download", return_value=_fake_yf_frame()):
        fetcher.fetch_ohlcv("msft", cache_dir=tmp_path)
        assert (tmp_path / "MSFT_ohlcv.parquet").exists()