"""
Tests for src.features.indicators.

These tests verify the indicator MATHS against hand-worked and published
examples. The RSI test in particular uses Wilder's canonical example 
an earlier implementation passed the easy edge cases but got this wrong 
(returned ~50.7 instead of ~70.53), which would have silently fed the model bad features.
This test exists so that bug can never come back unnoticed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.indicators import add_indicators, rsi, sma


# SMA 

def test_sma_basic_average() -> None:
    # Last 3 of [1,2,3,4,5] -> (3+4+5)/3 = 4.0
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(s, 3).iloc[-1] == 4.0


def test_sma_warmup_is_nan() -> None:
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert out.isna().sum() == 2  # first (window-1) values
    assert out.iloc[:2].isna().all()


def test_sma_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        sma(pd.Series([1.0, 2.0]), 0)


def test_sma_does_not_mutate_input() -> None:
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    original = s.copy()
    sma(s, 3)
    pd.testing.assert_series_equal(s, original)


# RSI 

def test_rsi_wilder_canonical_example() -> None:
    """Wilder's published 14-period example: first RSI value is ~70.53.

    Source: J. Welles Wilder, 'New Concepts in Technical Trading Systems' (1978).
    This is the regression guard for the seeding bug.
    """
    prices = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
    ]
    first_rsi = rsi(pd.Series(prices, dtype=float), 14).dropna().iloc[0]
    assert first_rsi == pytest.approx(70.53, abs=0.5)


def test_rsi_all_gains_is_100() -> None:
    """A strictly rising series has no losses -> RSI saturates at 100."""
    rising = pd.Series(range(1, 30), dtype=float)
    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_bounded_0_to_100() -> None:
    # A noisy but realistic-ish series should stay strictly in range.
    import numpy as np

    rng = np.random.default_rng(42)
    prices = pd.Series(100 + rng.standard_normal(200).cumsum())
    r = rsi(prices, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_warmup_is_nan() -> None:
    prices = pd.Series(range(1, 30), dtype=float)
    r = rsi(prices, 14)
    # Need `window` deltas before the first value; first valid at position 14.
    assert r.iloc[:14].isna().all()
    assert pd.notna(r.iloc[14])


def test_rsi_insufficient_data_all_nan() -> None:
    prices = pd.Series([1.0, 2.0, 3.0])  # fewer than window
    assert rsi(prices, 14).isna().all()


# add_indicators 

def test_add_indicators_attaches_columns_without_mutating() -> None:
    df = pd.DataFrame(
        {
            "Open": range(1, 60),
            "High": range(1, 60),
            "Low": range(1, 60),
            "Close": range(1, 60),
            "Volume": range(1, 60),
        },
        dtype=float,
    )
    original_cols = df.columns.tolist()
    out = add_indicators(df)
    # Input untouched
    assert df.columns.tolist() == original_cols
    # New columns present
    assert "sma_20" in out.columns
    assert "rsi_14" in out.columns


# Phase 2 indicators 

def test_macd_zero_on_constant_series() -> None:
    """On a flat price, fast and slow EMAs are equal, so every MACD part is 0."""
    from src.features.indicators import macd

    const = pd.Series([100.0] * 60)
    out = macd(const)
    assert out["macd"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert out["macd_signal"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert out["macd_hist"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_macd_rejects_fast_not_less_than_slow() -> None:
    from src.features.indicators import macd

    with pytest.raises(ValueError, match="must be <"):
        macd(pd.Series([1.0, 2.0, 3.0]), fast=26, slow=12)


def test_macd_hist_equals_line_minus_signal() -> None:
    from src.features.indicators import macd

    s = pd.Series(range(1, 100), dtype=float)
    out = macd(s)
    diff = out["macd"] - out["macd_signal"]
    pd.testing.assert_series_equal(out["macd_hist"], diff, check_names=False)


def test_bollinger_hand_verified() -> None:
    """Window=3 on [1,2,3,4,5]; at t=2 mid=2, population std=0.8165.

    upper = 2 + 2*0.8165 = 3.633, lower = 2 - 2*0.8165 = 0.367.
    """
    from src.features.indicators import bollinger_bands

    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = bollinger_bands(s, window=3, num_std=2.0)
    assert out["bb_mid"].iloc[2] == pytest.approx(2.0)
    assert out["bb_upper"].iloc[2] == pytest.approx(3.633, abs=0.01)
    assert out["bb_lower"].iloc[2] == pytest.approx(0.367, abs=0.01)


def test_bollinger_warmup_is_nan() -> None:
    from src.features.indicators import bollinger_bands

    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = bollinger_bands(s, window=3)
    assert out["bb_mid"].iloc[:2].isna().all()


def test_volume_ratio_hand_verified() -> None:
    """Volume [10,10,10,40], window=3: at t=3 avg of [10,10,40]? No — last 3 are
    indices 1,2,3 = [10,10,40], mean=20, ratio=40/20=2.0."""
    from src.features.indicators import volume_ratio

    v = pd.Series([10, 10, 10, 40], dtype=float)
    out = volume_ratio(v, window=3)
    assert out.iloc[3] == pytest.approx(2.0)


def test_momentum_hand_verified() -> None:
    """2-day momentum on [1,2,3,4,5] at t=2: 3/1 - 1 = 2.0."""
    from src.features.indicators import momentum

    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = momentum(s, periods=2)
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[:2].isna().all()


def test_momentum_rejects_bad_periods() -> None:
    from src.features.indicators import momentum

    with pytest.raises(ValueError, match="periods must be >= 1"):
        momentum(pd.Series([1.0, 2.0]), periods=0)


def test_add_indicators_includes_all_phase2_columns() -> None:
    df = pd.DataFrame(
        {
            "Open": range(1, 60),
            "High": range(1, 60),
            "Low": range(1, 60),
            "Close": range(1, 60),
            "Volume": range(1, 60),
        },
        dtype=float,
    )
    out = add_indicators(df)
    for col in [
        "macd", "macd_signal", "macd_hist", "bb_width",
        "vol_ratio_20", "momentum_1d", "momentum_5d", "momentum_20d",
    ]:
        assert col in out.columns, f"missing {col}"