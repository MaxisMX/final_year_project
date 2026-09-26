"""
Equity-curve figure for the backtest discussion (Sections 5.5 and 5.8).

Run from the project root:

    python -m src.equity_curve

Draws two panels on the same MSFT backtest, differing only in how much history
is available:

  Left   data cut off in mid-August 2026, approximating the run reported in
         Section 5.5.
  Right  all data available today, the run reported in Section 5.8.

Each panel shows the model-driven strategy, buy-and-hold, and the trivial
moving-average rule, all at a 0.1% cost per position change. Showing the two
side by side makes the window sensitivity described in Section 5.8 visible:
the code is identical, only the test period moves.

Saves equity_curves.png at report-ready resolution. Like wider_checks.py, this
script only reads data and never touches the web application.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # render to file without needing a display
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import backtest
from src.data.fetcher import fetch_ohlcv
from src.features.indicators import add_indicators
from src.features.target import BUY, SELL, attach_label, drop_unlabelled
from src.models.random_forest import (
    FEATURE_COLUMNS,
    chronological_split,
    train_random_forest,
)

warnings.filterwarnings("ignore")

TICKER = "MSFT"
AUGUST_CUTOFF = "2026-08-14"
SEPTEMBER_CUTOFF = "2026-09-22"
COST = 0.001

COLOURS = {
    "strategy": "#2f6f5e",  # matches the app's accent green
    "hold": "#55605a",
    "rule": "#b08a3e",
}


def run(end: str | None):
    """Backtest strategy, buy-and-hold and the rule on data up to `end`."""
    raw = fetch_ohlcv(TICKER, start="2015-01-01")
    if end is not None:
        raw = raw.loc[:end]  # cut BEFORE labelling, so no future leaks in

    clean = drop_unlabelled(
        attach_label(add_indicators(raw), horizon=5), horizon=5
    ).dropna(subset=FEATURE_COLUMNS)

    split = chronological_split(clean)
    model = train_random_forest(split.X_train, split.y_train)

    test = clean.loc[split.X_test.index]
    close = test["Close"]

    model_preds = pd.Series(model.predict(split.X_test), index=test.index)
    rule_preds = pd.Series(
        [BUY if c > s else SELL for c, s in zip(test["Close"], test["sma_20"])],
        index=test.index,
    )

    strat = backtest(close, model_preds, cost_per_trade=COST)
    rule = backtest(close, rule_preds, cost_per_trade=COST)
    hold_curve = (1.0 + close.pct_change().fillna(0.0)).cumprod()

    return strat, rule, hold_curve


def draw(ax, strat, rule, hold_curve, title):
    start = hold_curve.index[0].strftime("%b %Y")
    end = hold_curve.index[-1].strftime("%b %Y")

    ax.plot(strat.equity_curve, color=COLOURS["strategy"], lw=1.8,
            label=f"Model strategy ({strat.total_return:+.1%})")
    ax.plot(hold_curve, color=COLOURS["hold"], lw=1.4, ls="--",
            label=f"Buy & hold ({strat.bh_total_return:+.1%})")
    ax.plot(rule.equity_curve, color=COLOURS["rule"], lw=1.4,
            label=f"20-day average rule ({rule.total_return:+.1%})")

    ax.axhline(1.0, color="#cccccc", lw=0.8)
    ax.set_title(f"{title}\nTest period {start} to {end}", fontsize=11)
    ax.set_ylabel("Value of $1 invested")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(axis="y", color="#eeeeee")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)


def main():
    print("Running August-cutoff backtest...")
    aug = run(AUGUST_CUTOFF)
    print("Running backtest on all current data...")
    now = run(SEPTEMBER_CUTOFF)

    for label, (s, r, _) in (("August cutoff", aug), ("Current data", now)):
        print(f"\n{label}:")
        print(f"  Strategy     {s.total_return:+.1%}   ({s.n_trades} changes)")
        print(f"  Buy & hold   {s.bh_total_return:+.1%}")
        print(f"  Rule         {r.total_return:+.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    draw(axes[0], *aug, "Data to mid-August 2026")
    draw(axes[1], *now, "Data to September 2026")
    fig.tight_layout()
    fig.savefig("equity_curves.png", dpi=200, facecolor="white")
    print("\nSaved equity_curves.png")


if __name__ == "__main__":
    main()
    
