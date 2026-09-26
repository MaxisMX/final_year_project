"""
Plain-English explanation generator for BUY/SELL recommendations.

Phase 3 (part 1) the project's differentiator. Turns indicator values into
simple, friendly sentences a non-technical user can understand.

IMPORTANT distinction (This has been state in report)
------------------------------------------------
This module explains what the TECHNICAL INDICATORS are saying in plain language.
It does NOT claim to explain the model's internal reasoning. Those are different:
  - This generator = human-readable interpretation of the indicator signals.
  - SHAP (built next) = what actually drove the MODEL's specific decision.
Conflating the two would be an overclaim. We present this as "here's what the
signals mean," alongside the model's recommendation — not as "here's why the
neural network decided this."

Design
------
Each indicator has a small set of rules mapping its value to a short phrase and
a leaning (bullish / bearish / neutral). We collect the phrases, summarise the
overall picture, and pair it with the model's actual recommendation. Thresholds
use conventional technical-analysis levels (e.g. RSI > 70 = overbought), which
are documented so they can be defended in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.features.target import BUY, SELL


@dataclass
class Signal:
    """One indicator's plain-English reading."""

    indicator: str
    phrase: str
    leaning: str  # "bullish", "bearish", or "neutral"


@dataclass
class Explanation:
    """A full plain-English explanation paired with the model's recommendation."""

    recommendation: str  # "BUY" or "SELL"
    headline: str
    signals: list[Signal] = field(default_factory=list)

    def bullish_count(self) -> int:
        return sum(1 for s in self.signals if s.leaning == "bullish")

    def bearish_count(self) -> int:
        return sum(1 for s in self.signals if s.leaning == "bearish")

    def to_text(self) -> str:
        """Render the explanation as friendly multi-line text."""
        lines = [self.headline, ""]
        lines.append("Here's what the signals are showing:")
        for s in self.signals:
            lines.append(f"  \u2022 {s.phrase}")
        return "\n".join(lines)


# aIndividual indicator rules 
# Each returns a Signal. Thresholds use conventional TA levels; keep them here
# so they're documented and easy to defend/tune.

def _rsi_signal(rsi: float) -> Signal:
    if rsi >= 70:
        return Signal(
            "RSI",
            f"The stock looks overbought (RSI {rsi:.0f}) \u2014 it has risen sharply "
            "and may be due for a pause.",
            "bearish",
        )
    if rsi <= 30:
        return Signal(
            "RSI",
            f"The stock looks oversold (RSI {rsi:.0f}) \u2014 it has fallen sharply "
            "and may be due for a bounce.",
            "bullish",
        )
    return Signal(
        "RSI",
        f"Momentum is in a normal range (RSI {rsi:.0f}) \u2014 neither overbought "
        "nor oversold.",
        "neutral",
    )


def _macd_signal(macd_hist: float) -> Signal:
    if macd_hist > 0:
        return Signal(
            "MACD",
            "The trend indicator (MACD) is positive, suggesting upward momentum.",
            "bullish",
        )
    if macd_hist < 0:
        return Signal(
            "MACD",
            "The trend indicator (MACD) is negative, suggesting downward momentum.",
            "bearish",
        )
    return Signal("MACD", "The trend indicator (MACD) is flat.", "neutral")


def _momentum_signal(momentum_5d: float) -> Signal:
    pct = momentum_5d * 100
    if momentum_5d > 0.02:
        return Signal(
            "Momentum",
            f"The price has been climbing recently (up {pct:.1f}% over 5 days).",
            "bullish",
        )
    if momentum_5d < -0.02:
        return Signal(
            "Momentum",
            f"The price has been falling recently (down {abs(pct):.1f}% over 5 days).",
            "bearish",
        )
    return Signal(
        "Momentum",
        f"The price has been fairly steady recently ({pct:+.1f}% over 5 days).",
        "neutral",
    )


def _price_vs_average_signal(close: float, sma_20: float) -> Signal:
    if close > sma_20:
        return Signal(
            "Trend",
            "The price is above its recent average, a generally positive sign.",
            "bullish",
        )
    if close < sma_20:
        return Signal(
            "Trend",
            "The price is below its recent average, a generally cautious sign.",
            "bearish",
        )
    return Signal("Trend", "The price is right at its recent average.", "neutral")


def explain(
    recommendation: int,
    close: float,
    sma_20: float,
    rsi_14: float,
    macd_hist: float,
    momentum_5d: float,
) -> Explanation:
    """
    Produce a plain-English explanation alongside the model's recommendation.

    Args:
        recommendation: The model's prediction (BUY or SELL constant).
        close, sma_20, rsi_14, macd_hist, momentum_5d: indicator values for the day.

    Returns:
        An Explanation with a friendly headline and per-signal phrases.
    """
    if recommendation not in (BUY, SELL):
        raise ValueError(f"recommendation must be BUY ({BUY}) or SELL ({SELL}).")

    signals = [
        _price_vs_average_signal(close, sma_20),
        _rsi_signal(rsi_14),
        _macd_signal(macd_hist),
        _momentum_signal(momentum_5d),
    ]

    rec_word = "BUY" if recommendation == BUY else "SELL"
    if recommendation == BUY:
        headline = (
            "The model suggests this could be a BUY opportunity. "
            "Remember, this is not financial advice \u2014 here's the reasoning to "
            "help you judge for yourself:"
        )
    else:
        headline = (
            "The model suggests caution (SELL / stay out) for now. "
            "Remember, this is not financial advice \u2014 here's the reasoning to "
            "help you judge for yourself:"
        )

    return Explanation(recommendation=rec_word, headline=headline, signals=signals)


if __name__ == "__main__":
    # Demo against the latest row of cached MSFT data:
    #   python -m src.explain.plain_english
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.models.random_forest import FEATURE_COLUMNS, chronological_split, train_random_forest
    from src.features.target import attach_label, drop_unlabelled

    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    split = chronological_split(clean)
    model = train_random_forest(split.X_train, split.y_train)

    # Explain the most recent predictable day.
    last = clean.iloc[-1]
    pred = int(model.predict(last[FEATURE_COLUMNS].to_frame().T)[0])

    exp = explain(
        recommendation=pred,
        close=last["Close"],
        sma_20=last["sma_20"],
        rsi_14=last["rsi_14"],
        macd_hist=last["macd_hist"],
        momentum_5d=last["momentum_5d"],
    )
    print(f"Date: {clean.index[-1].date()}  |  Recommendation: {exp.recommendation}\n")
    print(exp.to_text())
    print(
        f"\n(Signal tally: {exp.bullish_count()} bullish, {exp.bearish_count()} bearish)"
    )
    print(
        "\nNote: these sentences interpret the INDICATORS in plain language. They\n"
        "are shown alongside the model's recommendation, not as the model's own\n"
        "internal reasoning (that's what SHAP, built next, addresses)."
    )