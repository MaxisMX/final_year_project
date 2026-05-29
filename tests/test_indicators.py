"""Tests for src.features.indicators.

These tests verify the indicator MATHS against hand-worked and published
examples. The RSI test in particular uses Wilder's canonical example — an
earlier implementation passed the easy edge cases but got this wrong (returned
~50.7 instead of ~70.53), which would have silently fed the model bad features.
This test exists so that bug can never come back unnoticed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.indicators import add_indicators, rsi, sma


# --- SMA ---------------------------------------------------------------------

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


# --- RSI ---------------------------------------------------------------------

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


# --- add_indicators ----------------------------------------------------------

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