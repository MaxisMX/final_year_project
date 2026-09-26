"""Wider evaluation checks for the final report (Section 5.8).

Run from the project root:

    python wider_checks.py

This script only READS data and prints tables. It does not import or modify the
web service, the Flask app, or any saved model, so it cannot affect the running
application.

What it produces

  Table A  Random Forest walk-forward accuracy vs majority baseline across ten
           stocks spanning several sectors, rather than the three used in
           Section 5.3.
  Table B  The MSFT backtest repeated at four transaction-cost levels, showing
           how sensitive the result is to the cost assumption.
  Table C  A trivial rule (hold when the price is above its 20-day average)
           backtested on the same data, as a baseline that is not a
           majority-class classifier.
  Fold dates for MSFT, so the report can state which market periods the
           walk-forward folds actually cover.

Only the Random Forest is used. The three neural models are slow and, as
Section 5.4 shows, collapse to a single class in many folds; re-running them on
more tickers would add computation without adding evidence.
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.backtest.engine import backtest
from src.backtest.walk_forward import walk_forward_validate
from src.data.fetcher import fetch_ohlcv
from src.features.indicators import add_indicators
from src.features.target import BUY, SELL, attach_label, drop_unlabelled
from src.models.random_forest import (
    FEATURE_COLUMNS,
    chronological_split,
    train_random_forest,
)

warnings.filterwarnings("ignore")

# Ten tickers across sectors and volatility profiles. SPY, MSFT and TSLA are
# kept so the wider run can be compared directly with Section 5.3.
TICKERS = [
    ("SPY", "Broad market index"),
    ("MSFT", "Large-cap technology"),
    ("TSLA", "High-volatility growth"),
    ("AAPL", "Large-cap technology"),
    ("JNJ", "Healthcare, defensive"),
    ("JPM", "Financials"),
    ("XOM", "Energy"),
    ("KO", "Consumer staples"),
    ("NVDA", "Semiconductors"),
    ("WMT", "Retail"),
]

COST_LEVELS = [0.0, 0.0005, 0.001, 0.002]


def prepare(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Fetch and build the labelled, NaN-free feature frame for one ticker."""
    raw = fetch_ohlcv(ticker, start=start)
    feat = add_indicators(raw)
    labelled = attach_label(feat, horizon=5)
    return drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)


def table_a() -> pd.DataFrame:
    """Random Forest walk-forward accuracy vs baseline across the ten tickers."""
    rows = []
    for ticker, sector in TICKERS:
        try:
            clean = prepare(ticker)
            result = walk_forward_validate(clean, n_splits=5)
            rows.append(
                {
                    "Ticker": ticker,
                    "Sector": sector,
                    "Accuracy": round(result.mean_accuracy(), 3),
                    "SD": round(result.std_accuracy(), 3),
                    "Baseline": round(result.mean_majority_baseline(), 3),
                    "Beats?": "Yes" if result.beats_baseline_on_average() else "No",
                }
            )
            print(f"  {ticker:6} done")
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't stop the run
            print(f"  {ticker:6} FAILED: {e}")
    return pd.DataFrame(rows)


def _model_predictions(clean: pd.DataFrame):
    """Train RF on the earlier period and predict the held-out period."""
    split = chronological_split(clean)
    model = train_random_forest(split.X_train, split.y_train)
    preds = pd.Series(model.predict(split.X_test), index=split.X_test.index)
    close = clean.loc[split.X_test.index, "Close"]
    return preds, close


def _rule_predictions(clean: pd.DataFrame):
    """Trivial rule: hold the stock when its close is above its 20-day average."""
    split = chronological_split(clean)
    test = clean.loc[split.X_test.index]
    preds = pd.Series(
        [BUY if c > s else SELL for c, s in zip(test["Close"], test["sma_20"])],
        index=test.index,
    )
    return preds, test["Close"]


def table_b_and_c(ticker: str = "MSFT") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cost sensitivity for the model, and the same for the trivial rule."""
    clean = prepare(ticker)
    model_preds, close = _model_predictions(clean)
    rule_preds, _ = _rule_predictions(clean)

    b_rows, c_rows = [], []
    for cost in COST_LEVELS:
        m = backtest(close, model_preds, cost_per_trade=cost)
        r = backtest(close, rule_preds, cost_per_trade=cost)
        b_rows.append(
            {
                "Cost per change": f"{cost:.2%}",
                "Strategy return": f"{m.total_return:.1%}",
                "Buy & hold": f"{m.bh_total_return:.1%}",
                "Sharpe": round(m.sharpe, 2),
                "Max drawdown": f"{m.max_drawdown:.1%}",
                "Changes": m.n_trades,
            }
        )
        c_rows.append(
            {
                "Cost per change": f"{cost:.2%}",
                "Rule return": f"{r.total_return:.1%}",
                "Buy & hold": f"{r.bh_total_return:.1%}",
                "Sharpe": round(r.sharpe, 2),
                "Max drawdown": f"{r.max_drawdown:.1%}",
                "Changes": r.n_trades,
            }
        )
    return pd.DataFrame(b_rows), pd.DataFrame(c_rows)


def fold_dates(ticker: str = "MSFT") -> pd.DataFrame:
    """The date range each walk-forward fold tests on, for the regime discussion."""
    clean = prepare(ticker)
    result = walk_forward_validate(clean, n_splits=5)
    return pd.DataFrame(
        [
            {
                "Fold": f.fold,
                "Test start": f.test_start.date(),
                "Test end": f.test_end.date(),
                "Accuracy": round(f.accuracy, 3),
                "Baseline": round(f.majority_baseline_accuracy, 3),
            }
            for f in result.folds
        ]
    )


def main() -> None:
    print("Running wider evaluation checks. This takes a few minutes.\n")

    print("TABLE A - Random Forest across ten stocks")
    a = table_a()
    print("\n" + a.to_string(index=False))
    beat = (a["Beats?"] == "Yes").sum() if len(a) else 0
    print(f"\n  Beat the baseline on {beat} of {len(a)} stocks.\n")

    print("TABLE B - MSFT backtest, sensitivity to transaction costs")
    b, c = table_b_and_c()
    print("\n" + b.to_string(index=False) + "\n")

    print("TABLE C - Trivial rule (price above 20-day average), same data")
    print("\n" + c.to_string(index=False) + "\n")

    print("FOLD DATES - which market periods the MSFT folds cover")
    print("\n" + fold_dates().to_string(index=False) + "\n")

    a.to_csv("wider_checks_table_a.csv", index=False)
    b.to_csv("wider_checks_table_b.csv", index=False)
    c.to_csv("wider_checks_table_c.csv", index=False)
    print("Saved: wider_checks_table_a.csv, _b.csv, _c.csv")


if __name__ == "__main__":
    main()
