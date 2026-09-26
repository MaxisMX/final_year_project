"""
Tests for src.explain.plain_english.

We verify that each indicator rule fires at the right thresholds and leans the
right way (bullish/bearish/neutral), and that the explanation pairs correctly
with the recommendation. These are behavioural tests the 'maths' here is the
threshold logic, so we check the boundaries.
"""

from __future__ import annotations

import pytest

from src.explain.plain_english import (
    _macd_signal,
    _momentum_signal,
    _price_vs_average_signal,
    _rsi_signal,
    explain,
)
from src.features.target import BUY, SELL


# RSI thresholds 

def test_rsi_overbought_is_bearish() -> None:
    assert _rsi_signal(75).leaning == "bearish"
    assert _rsi_signal(70).leaning == "bearish"  # boundary


def test_rsi_oversold_is_bullish() -> None:
    assert _rsi_signal(25).leaning == "bullish"
    assert _rsi_signal(30).leaning == "bullish"  # boundary


def test_rsi_normal_is_neutral() -> None:
    assert _rsi_signal(50).leaning == "neutral"


# MACD 

def test_macd_positive_is_bullish() -> None:
    assert _macd_signal(0.5).leaning == "bullish"


def test_macd_negative_is_bearish() -> None:
    assert _macd_signal(-0.5).leaning == "bearish"


def test_macd_zero_is_neutral() -> None:
    assert _macd_signal(0.0).leaning == "neutral"


# Momentum 

def test_momentum_strong_up_is_bullish() -> None:
    assert _momentum_signal(0.05).leaning == "bullish"


def test_momentum_strong_down_is_bearish() -> None:
    assert _momentum_signal(-0.05).leaning == "bearish"


def test_momentum_flat_is_neutral() -> None:
    assert _momentum_signal(0.0).leaning == "neutral"


# Price vs average 

def test_price_above_average_is_bullish() -> None:
    assert _price_vs_average_signal(110, 100).leaning == "bullish"


def test_price_below_average_is_bearish() -> None:
    assert _price_vs_average_signal(90, 100).leaning == "bearish"


# Full explanation 

def test_explain_buy_recommendation() -> None:
    exp = explain(
        recommendation=BUY,
        close=110,
        sma_20=100,
        rsi_14=25,
        macd_hist=0.5,
        momentum_5d=0.05,
    )
    assert exp.recommendation == "BUY"
    assert "BUY" in exp.headline
    assert len(exp.signals) == 4
    # All four signals here are bullish.
    assert exp.bullish_count() == 4


def test_explain_sell_recommendation() -> None:
    exp = explain(
        recommendation=SELL,
        close=90,
        sma_20=100,
        rsi_14=75,
        macd_hist=-0.5,
        momentum_5d=-0.05,
    )
    assert exp.recommendation == "SELL"
    assert exp.bearish_count() == 4


def test_explain_includes_not_advice_disclaimer() -> None:
    exp = explain(BUY, 110, 100, 50, 0.1, 0.01)
    assert "not financial advice" in exp.headline.lower()


def test_explain_to_text_lists_all_signals() -> None:
    exp = explain(BUY, 110, 100, 50, 0.1, 0.01)
    text = exp.to_text()
    for s in exp.signals:
        assert s.phrase in text


def test_explain_rejects_invalid_recommendation() -> None:
    with pytest.raises(ValueError, match="must be BUY"):
        explain(99, 110, 100, 50, 0.1, 0.01)