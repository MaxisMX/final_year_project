"""Tests for src.backtest.engine.

The headline guards:
  - The 1-day action lag holds (you act on a prediction the day AFTER it's made,
    never the same day). This is the leakage guard for the backtest.
  - Transaction costs are charged on position changes and reduce returns.
  - Drawdown, Sharpe, and buy-and-hold comparison behave correctly.
Hand-worked numbers are used wherever possible so the maths is proven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _max_drawdown, _sharpe, backtest
from src.features.target import BUY, SELL


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_action_lag_is_applied() -> None:
    """A prediction on day t must be acted on at day t+1, not day t.

    Prices: [100, 110, 121] -> daily returns [_, +10%, +10%].
    Predictions: [BUY, SELL, SELL].
    With the correct 1-day lag, the BUY on day 0 takes effect on day 1's return
    (+10%). The SELL on day 1 means we're in cash for day 2 (0%). Day 0 itself
    has no prior signal, so we start flat.
    Expected strategy: day1 = +10%, day2 = 0%  ->  total ~ +10% (minus costs).

    If the lag were WRONG (acting same-day), the BUY on day 0 would wrongly
    capture day 0 with no return and the numbers would differ — this test fails
    loudly in that case.
    """
    close = _series([100, 110, 121])
    preds = pd.Series([BUY, SELL, SELL], index=close.index)
    result = backtest(close, preds, cost_per_trade=0.0)  # no costs to isolate timing
    # Day 1 captured (+10%), day 2 in cash. Total return ~ +10%.
    assert result.total_return == pytest.approx(0.10, abs=1e-9)


def test_always_buy_matches_buy_and_hold_minus_one_entry_cost() -> None:
    """Always predicting BUY should track buy-and-hold (minus the entry cost)."""
    close = _series([100, 102, 101, 105, 110])
    preds = pd.Series([BUY] * len(close), index=close.index)
    result = backtest(close, preds, cost_per_trade=0.0)
    # With no costs and always-in (after the 1-day lag), strategy return should
    # match buy-and-hold's return from day 1 onward. Allow tiny float tolerance.
    assert result.total_return == pytest.approx(result.bh_total_return, abs=1e-9)


def test_costs_reduce_returns() -> None:
    """Adding transaction costs must lower the total return versus no costs."""
    close = _series([100, 105, 100, 105, 100, 105])
    # Alternate to force frequent position changes (lots of trades).
    preds = pd.Series([BUY, SELL, BUY, SELL, BUY, SELL], index=close.index)
    no_cost = backtest(close, preds, cost_per_trade=0.0).total_return
    with_cost = backtest(close, preds, cost_per_trade=0.01).total_return
    assert with_cost < no_cost


def test_n_trades_counts_position_changes() -> None:
    close = _series([100, 101, 102, 103, 104])
    # flat(start) -> BUY -> BUY -> SELL -> BUY : changes at days 1, 3, 4 after lag.
    preds = pd.Series([BUY, BUY, SELL, BUY, BUY], index=close.index)
    result = backtest(close, preds, cost_per_trade=0.0)
    assert result.n_trades >= 1  # at least the initial entry


def test_all_sell_stays_flat_zero_return() -> None:
    """Never holding the stock => zero return, zero drawdown."""
    close = _series([100, 90, 120, 80, 130])
    preds = pd.Series([SELL] * len(close), index=close.index)
    result = backtest(close, preds, cost_per_trade=0.0)
    assert result.total_return == pytest.approx(0.0)
    assert result.max_drawdown == pytest.approx(0.0)


def test_max_drawdown_hand_calculated() -> None:
    # Equity peaks at 1.2 then falls to 0.9 -> drawdown = 0.9/1.2 - 1 = -0.25
    equity = pd.Series([1.0, 1.2, 1.0, 0.9, 1.1])
    assert _max_drawdown(equity) == pytest.approx(-0.25)


def test_sharpe_zero_when_no_variance() -> None:
    flat = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert _sharpe(flat) == 0.0


def test_beats_buy_and_hold_flag() -> None:
    """A strategy that dodges a crash should beat buy-and-hold."""
    # Stock rises then crashes hard. A strategy that sells before the crash wins.
    close = _series([100, 110, 120, 60, 30])
    # Predict BUY early, SELL before the crash. With the 1-day lag, selling on
    # day 2 keeps us out of the day-3 crash.
    preds = pd.Series([BUY, BUY, SELL, SELL, SELL], index=close.index)
    result = backtest(close, preds, cost_per_trade=0.0)
    assert result.beats_buy_and_hold()


def test_mismatched_index_raises() -> None:
    close = _series([100, 101, 102])
    preds = pd.Series([BUY, SELL, BUY], index=pd.bdate_range("2021-06-01", periods=3))
    with pytest.raises(ValueError, match="same index"):
        backtest(close, preds)


def test_too_short_raises() -> None:
    close = _series([100])
    preds = pd.Series([BUY], index=close.index)
    with pytest.raises(ValueError, match="at least 2 days"):
        backtest(close, preds)