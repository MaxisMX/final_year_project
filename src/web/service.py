"""Service layer: ticker -> recommendation + explanation of that recommendation.

Phase 5. Rewritten from the Phase 4 version in response to feedback on the draft
report.

What changed, and why
---------------------
The earlier version served an LSTM verdict and displayed a rule-based reading of
the indicators beneath it. Those two were produced independently: the sentences
would have been identical had the model predicted the opposite class, and SHAP —
the one component that genuinely explains a model — was applied to the Random
Forest, which was not the model being served. The explanation therefore sat
ALONGSIDE the recommendation rather than explaining it.

This version closes that gap:

  1. The Random Forest is the served model, so the model being explained and the
     model making the recommendation are the same one. No model in the
     comparison beat a majority-class baseline, so no predictive advantage is
     being given up by choosing the interpretable one.
  2. SHAP is computed for the exact prediction the user sees, and its
     attributions are rendered as plain English by explain/shap_english.py.
  3. Before any explanation is generated, models/confidence.py checks whether
     the model is behaving like a predictor at all. A collapsed model gets no
     explanation — only a statement that no reliable signal is available.

Train/test handling
-------------------
The model is trained on the earlier portion of the history via
chronological_split, and both the confidence check and the served prediction use
the held-out later portion. This keeps the user-facing verdict genuinely
out-of-sample and consistent with the evaluation in Chapter 5. The trade-off is
that the model does not learn from the most recent stretch of data; a production
system would retrain on everything and validate separately.

Caching
-------
Random Forest trains in seconds rather than the tens of seconds the LSTM needed,
so caching matters far less than it did. It is kept because repeated requests for
the same ticker on the same day should not repeat the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.fetcher import fetch_ohlcv
from src.explain.shap_analysis import compute_shap
from src.explain.shap_english import ModelExplanation, explain_from_shap
from src.features.indicators import add_indicators
from src.features.target import BUY, attach_label, drop_unlabelled
from src.models.confidence import ConfidenceReport, assess_confidence
from src.models.random_forest import (
    FEATURE_COLUMNS,
    chronological_split,
    train_random_forest,
)

# In-memory cache: (ticker, latest date) -> trained model. Cleared on restart.
_MODEL_CACHE: dict[tuple[str, str], RandomForestClassifier] = {}


@dataclass
class Recommendation:
    """Everything the web app needs to display for one ticker."""

    ticker: str
    as_of_date: str
    recommendation: str  # "BUY" or "SELL"
    confidence: ConfidenceReport
    # None when the model has collapsed: there is no reasoning to describe.
    explanation: ModelExplanation | None = None
    price_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable form for the API."""
        payload = {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "recommendation": self.recommendation,
            "confidence": self.confidence.to_dict(),
            "explained": self.explanation is not None,
            "price_history": self.price_history,
        }
        if self.explanation is not None:
            payload["headline"] = self.explanation.headline
            payload["caveat"] = self.explanation.caveat
            payload["signals"] = [
                {
                    "feature": s.feature,
                    "display_name": s.display_name,
                    "shap_value": round(s.shap_value, 4),
                    "direction": s.direction,
                    "sentence": s.sentence(),
                }
                for s in self.explanation.signals
            ]
        else:
            payload["headline"] = self.confidence.message
            payload["caveat"] = ""
            payload["signals"] = []
        return payload


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
    history_days: int = 120,
    n_signals: int = 3,
    use_cache: bool = True,
) -> Recommendation:
    """Produce a recommendation and an explanation of it for a ticker.

    Args:   
        ticker: Stock symbol (e.g. "MSFT").
        start: Earliest date of training data.
        history_days: How many recent days of price to return for the chart.
        n_signals: How many SHAP-ranked factors to describe.
        use_cache: Reuse a previously trained model for this ticker if available.

    Returns:
        A Recommendation. When the model has collapsed, `explanation` is None and
        `confidence.message` carries the reason.

    Raises:
        TickerError: if the ticker is invalid or has too little data.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise TickerError("Ticker must not be empty.")

    clean = _prepare_data(ticker, start)

    if len(clean) < 250:
        raise TickerError(
            f"'{ticker}' has too little usable history ({len(clean)} rows) to "
            "train and assess a model. Around a year of trading days is needed."
        )

    split = chronological_split(clean)

    cache_key = (ticker, str(clean.index.max().date()))
    if use_cache and cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
    else:
        model = train_random_forest(split.X_train, split.y_train)
        _MODEL_CACHE[cache_key] = model

    # Held-out rows: the served prediction is the most recent of these.
    X_eval = split.X_test
    latest_date = X_eval.index[-1]
    latest_X = X_eval.iloc[[-1]]
    latest_pred = int(model.predict(latest_X)[0])

    # Is this model behaving like a predictor, or defaulting to one class?
    confidence = assess_confidence(model, X_eval)

    explanation: ModelExplanation | None = None
    if confidence.should_explain:
        shap_values = compute_shap(model, latest_X).shap_values
        explanation = explain_from_shap(
            shap_row=shap_values.iloc[-1],
            indicator_row=clean.loc[latest_date],
            recommendation=latest_pred,
            n_signals=n_signals,
        )

    recent = clean.loc[:latest_date].iloc[-history_days:]
    price_history = [
        {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
        for d, c in recent["Close"].items()
    ]

    return Recommendation(
        ticker=ticker,
        as_of_date=latest_date.strftime("%Y-%m-%d"),
        recommendation="BUY" if latest_pred == BUY else "SELL",
        confidence=confidence,
        explanation=explanation,
        price_history=price_history,
    )


def clear_cache() -> None:
    """Clear the in-memory model cache (useful for tests and fresh runs)."""
    _MODEL_CACHE.clear()


if __name__ == "__main__":
    # Demo: python -m src.web.service MSFT
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    print(f"Getting recommendation for {ticker}...")
    rec = get_recommendation(ticker)

    print(f"\nTicker: {rec.ticker}  |  As of: {rec.as_of_date}")
    print(f"Recommendation: {rec.recommendation}")
    print(f"Confidence: {rec.confidence.status.value}\n")

    if rec.explanation is not None:
        print(rec.explanation.to_text())
        if rec.confidence.status.value == "low_confidence":
            print(f"\n{rec.confidence.message}")
    else:
        print(rec.confidence.message)

    print(f"\nPrice history points for chart: {len(rec.price_history)}")