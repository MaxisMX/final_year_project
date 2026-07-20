"""Service layer: ticker -> recommendation + plain-English explanation.

Phase 4 (part 1). This is the single entry point the web app calls. It wires
together everything built so far:
    fetch data -> indicators -> label -> train LSTM -> predict latest day
    -> plain-English explanation -> price data for the chart.

Kept as pure Python (no Flask here) so the core logic is fully testable before
any web complexity is added.

Train-on-demand + caching
-------------------------
Per the chosen design, each ticker's LSTM is trained on demand. Because that is
slow (tens of seconds), we cache the trained artifacts per ticker in memory: the
first request for a ticker trains; later requests reuse the cached model. A
production system would instead pre-train and persist models — noted as a
trade-off in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


import pandas as pd

from src.data.fetcher import fetch_ohlcv
from src.explain.plain_english import Explanation, explain
from src.features.indicators import add_indicators
from src.features.target import attach_label, drop_unlabelled
from src.models.lstm import LSTMArtifacts, predict_lstm, train_lstm
from src.models.random_forest import FEATURE_COLUMNS

# In-memory cache: ticker -> trained LSTM artifacts. Cleared on process restart.
_MODEL_CACHE: dict[str, LSTMArtifacts] = {}


@dataclass
class Recommendation:
    """Everything the web app needs to display for one ticker."""

    ticker: str
    as_of_date: str
    recommendation: str  # "BUY" or "SELL"
    explanation: Explanation
    # Recent price history for the chart: list of {date, close}.
    price_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable form for the API."""
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "recommendation": self.recommendation,
            "headline": self.explanation.headline,
            "signals": [
                {"indicator": s.indicator, "phrase": s.phrase, "leaning": s.leaning}
                for s in self.explanation.signals
            ],
            "bullish_count": self.explanation.bullish_count(),
            "bearish_count": self.explanation.bearish_count(),
            "price_history": self.price_history,
        }


class TickerError(ValueError):
    """Raised when a ticker can't be fetched or has insufficient data."""



def _prepare_data(ticker: str, start: str) -> pd.DataFrame:
    """Fetch and build the labelled, feature-rich, clean frame for a ticker."""
    try:
        raw = fetch_ohlcv(ticker, start=start)
    except ValueError as e:
        raise TickerError(
            f"Could not fetch data for '{ticker}'. Check the symbol is valid "
            f"(e.g. AAPL, MSFT). Original error: {e}"
        ) from e

    feat = add_indicators(raw)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)
    return clean


def get_recommendation(
    ticker: str,
    start: str = "2015-01-01",
    lookback: int = 60,
    history_days: int = 120,
    use_cache: bool = True,
) -> Recommendation:
    """Produce a BUY/SELL recommendation with explanation for a ticker.

    Args:
        ticker: Stock symbol (e.g. "MSFT").
        start: Earliest date of training data.
        lookback: LSTM sequence window.
        history_days: How many recent days of price to return for the chart.
        use_cache: Reuse a previously trained LSTM for this ticker if available.

    Returns:
        A Recommendation with the latest prediction, explanation, and price data.

    Raises:
        TickerError: if the ticker is invalid or has too little data.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise TickerError("Ticker must not be empty.")

    clean = _prepare_data(ticker, start)

    if len(clean) < lookback + 10:
        raise TickerError(
            f"'{ticker}' has too little usable history ({len(clean)} rows) to "
            f"train a model with a {lookback}-day lookback."
        )

    # Train (or reuse cached) LSTM for this ticker.
    cache_key = (ticker, str(clean.index.max().date()))
    if use_cache and cache_key in _MODEL_CACHE:
        artifacts = _MODEL_CACHE[cache_key]
    else:
        artifacts = train_lstm(clean, lookback=lookback)
        _MODEL_CACHE[cache_key] = artifacts

    # Predict on the most recent window. predict_lstm returns predictions aligned
    # to each window's last day; we want the very latest one.
    preds, _ = predict_lstm(artifacts, clean)
    latest_pred = int(preds.iloc[-1])
    latest_date = preds.index[-1]

    # Indicator values for the latest day, for the plain-English explanation.
    latest_row = clean.loc[latest_date]
    explanation = explain(
        recommendation=latest_pred,
        close=float(latest_row["Close"]),
        sma_20=float(latest_row["sma_20"]),
        rsi_14=float(latest_row["rsi_14"]),
        macd_hist=float(latest_row["macd_hist"]),
        momentum_5d=float(latest_row["momentum_5d"]),
    )

    # Recent price history for the chart.
    recent = clean.iloc[-history_days:]
    price_history = [
        {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
        for d, c in recent["Close"].items()
    ]

    from src.features.target import BUY

    return Recommendation(
        ticker=ticker,
        as_of_date=latest_date.strftime("%Y-%m-%d"),
        recommendation="BUY" if latest_pred == BUY else "SELL",
        explanation=explanation,
        price_history=price_history,
    )


def clear_cache() -> None:
    """Clear the in-memory model cache (useful for tests and fresh runs)."""
    _MODEL_CACHE.clear()


if __name__ == "__main__":
    # Demo: python -m src.web.service
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    print(f"Getting recommendation for {ticker} (training LSTM on demand)...")
    rec = get_recommendation(ticker)
    print(f"\nTicker: {rec.ticker}  |  As of: {rec.as_of_date}")
    print(f"Recommendation: {rec.recommendation}\n")
    print(rec.explanation.to_text())
    print(f"\nPrice history points for chart: {len(rec.price_history)}")