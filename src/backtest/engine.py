"""Backtest a BUY/SELL strategy against a buy-and-hold baseline.

Phase 1.5 deliverable — the last piece of the thin end-to-end slice.

The honest question this answers
--------------------------------
Accuracy is not profit. A model can be 'right' often and still lose money if it
is wrong on the big moves. This backtest asks: if you had actually traded on the
model's predictions, would you have beaten simply buying and holding the stock?

How the strategy works
----------------------
  - The model predicts BUY or SELL for each day using that day's close.
  - You can only ACT on the next day: when BUY, you hold the stock for the next
    day's return; when SELL, you are in cash (0 return) for the next day.
  - This 1-day lag is essential. Trading at the same close you used to predict
    would be using information you did not have yet — a subtle leakage that
    silently inflates returns. We shift positions forward by one day and test it.

Transaction costs
-----------------
Every time the position CHANGES (cash->stock or stock->cash), a cost is charged
(default 0.1% of the traded amount). Costs are what kill most naive strategies:
a model that flips position constantly bleeds money on fees even if its
directional calls are slightly better than chance.

Metrics (as named in the project brief)
----------------------------------------
  - Total Return: overall growth over the test period.
  - Sharpe Ratio: return per unit of risk (annualised). Higher is better.
  - Max Drawdown: the worst peak-to-trough fall. Less negative is better.
These are reported for BOTH the strategy and buy-and-hold so the comparison is
explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.target import BUY

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Performance of one strategy run, with buy-and-hold for comparison."""

    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    # Buy-and-hold baseline over the same period:
    bh_total_return: float
    bh_sharpe: float
    bh_max_drawdown: float
    equity_curve: pd.Series  # strategy cumulative value, starts at 1.0

    def beats_buy_and_hold(self) -> bool:
        return self.total_return > self.bh_total_return


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of an equity curve (returned as a negative)."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _sharpe(daily_returns: pd.Series) -> float:
    """Annualised Sharpe ratio (risk-free rate assumed 0 for simplicity).

    Returns 0.0 if returns have no variance (avoids divide-by-zero).
    """
    if daily_returns.std() == 0 or len(daily_returns) == 0:
        return 0.0
    return float(
        np.sqrt(TRADING_DAYS_PER_YEAR) * daily_returns.mean() / daily_returns.std()
    )


def backtest(
    close: pd.Series,
    predictions: pd.Series,
    cost_per_trade: float = 0.001,
) -> BacktestResult:
    """Run a long/flat backtest of predictions against buy-and-hold.

    Args:
        close: Closing prices over the test period, date-indexed.
        predictions: BUY/SELL predictions aligned to the SAME dates as `close`.
            A prediction made on day t is acted on at day t+1's return.
        cost_per_trade: Fractional cost charged whenever the position changes
            (0.001 = 0.1%).

    Returns:
        BacktestResult with strategy and buy-and-hold metrics.
    """
    if not close.index.equals(predictions.index):
        raise ValueError("close and predictions must share the same index.")
    if len(close) < 2:
        raise ValueError("Need at least 2 days to backtest.")

    # Daily returns of the underlying stock.
    daily_ret = close.pct_change().fillna(0.0)

    # Position: 1 when the model said BUY, 0 (cash) otherwise.
    # CRITICAL: shift forward by 1 day. The prediction made using day t's close
    # can only be acted on starting day t+1. The first day has no prior signal,
    # so position starts flat (0).
    position = (predictions == BUY).astype(int).shift(1).fillna(0)

    # Cost is charged on the days the position changes.
    position_changes = position.diff().abs().fillna(position.abs())
    costs = position_changes * cost_per_trade

    # Strategy daily return = position * stock return, minus costs that day.
    strategy_ret = position * daily_ret - costs

    # Equity curves (start at 1.0).
    strat_equity = (1.0 + strategy_ret).cumprod()
    bh_equity = (1.0 + daily_ret).cumprod()

    return BacktestResult(
        total_return=float(strat_equity.iloc[-1] - 1.0),
        sharpe=_sharpe(strategy_ret),
        max_drawdown=_max_drawdown(strat_equity),
        n_trades=int(position_changes.sum()),
        bh_total_return=float(bh_equity.iloc[-1] - 1.0),
        bh_sharpe=_sharpe(daily_ret),
        bh_max_drawdown=_max_drawdown(bh_equity),
        equity_curve=strat_equity,
    )


if __name__ == "__main__":
    # Full thin-slice backtest against cached MSFT data:
    #   python -m src.backtest.engine
    import pandas as pd

    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled
    from src.models.random_forest import (
        FEATURE_COLUMNS,
        chronological_split,
        train_random_forest,
    )

    print("Building features, training model, generating test-set predictions...")
    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    split = chronological_split(clean)
    model = train_random_forest(split.X_train, split.y_train)

    preds = pd.Series(model.predict(split.X_test), index=split.X_test.index)
    test_close = clean.loc[split.X_test.index, "Close"]

    result = backtest(test_close, preds, cost_per_trade=0.001)

    print("\n" + "=" * 55)
    print(f"BACKTEST: MSFT test period from {test_close.index[0].date()}")
    print("=" * 55)
    print(f"{'':22}{'Strategy':>14}{'Buy & Hold':>16}")
    print(f"{'Total return':22}{result.total_return:>13.1%}{result.bh_total_return:>16.1%}")
    print(f"{'Sharpe ratio':22}{result.sharpe:>13.2f}{result.bh_sharpe:>16.2f}")
    print(f"{'Max drawdown':22}{result.max_drawdown:>13.1%}{result.bh_max_drawdown:>16.1%}")
    print(f"{'Number of trades':22}{result.n_trades:>13}")
    print("=" * 55)
    print(f"Beats buy-and-hold? {'YES' if result.beats_buy_and_hold() else 'NO'}")
    print(
        "\nReminder: beating buy-and-hold is genuinely hard. A well-measured 'NO'\n"
        "is an honest, reportable finding — not a failure of the project."
    )