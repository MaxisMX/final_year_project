"""Plain-English explanations derived from the model's own SHAP attributions.

Phase 5 — the fix for the limitation identified in the draft report.

What changed and why
--------------------
The original design had two disconnected layers: a rule-based generator that
read indicator values, and SHAP analysis that explained the Random Forest. The
web app served an LSTM verdict. So the explanation shown to the user explained
neither the deployed model nor any model at all — it described the indicators
and was placed next to an unrelated prediction.

This module closes that gap. It takes the SHAP values computed for the SAME
prediction the user is shown, ranks the features by how strongly each pushed
that specific decision, and renders the top few as sentences a non-technical
reader can follow. The explanation is therefore an explanation OF the verdict,
not merely one displayed ALONGSIDE it.

The honest limit
----------------
SHAP faithfully reports what a model did. It says nothing about whether the
model is any good. A collapsed model — one predicting the same class every day —
still produces SHAP attributions, and rendering them as fluent prose would
manufacture a rationale for what is really a default. Collapse detection
therefore belongs upstream of this module: when the model has collapsed, no
explanation should be generated at all. See models/confidence.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.features.target import BUY, SELL

# Human-readable names for the ten feature columns.
DISPLAY_NAMES: dict[str, str] = {
    "sma_20": "Price vs 20-day average",
    "rsi_14": "RSI (momentum)",
    "macd": "MACD line",
    "macd_signal": "MACD signal line",
    "macd_hist": "MACD histogram",
    "bb_width": "Bollinger Band width",
    "vol_ratio_20": "Trading volume",
    "momentum_1d": "1-day price change",
    "momentum_5d": "5-day price change",
    "momentum_20d": "20-day price change",
}


@dataclass
class DrivenSignal:
    """One feature's contribution to the model's decision, in plain English."""

    feature: str
    display_name: str
    shap_value: float
    reading: str  # what the indicator itself is showing
    push: str  # how it moved the model, in words

    @property
    def direction(self) -> str:
        if self.shap_value > 0:
            return "toward BUY"
        if self.shap_value < 0:
            return "toward SELL"
        return "neither way"

    def sentence(self) -> str:
        return f"{self.reading} {self.push}"


@dataclass
class ModelExplanation:
    """A plain-English explanation of one model prediction."""

    recommendation: str  # "BUY" or "SELL"
    headline: str
    signals: list[DrivenSignal] = field(default_factory=list)
    caveat: str = ""

    def to_text(self) -> str:
        lines = [self.headline, ""]
        lines.append("What drove this particular call:")
        for s in self.signals:
            lines.append(f"  \u2022 {s.sentence()}")
        if self.caveat:
            lines.extend(["", self.caveat])
        return "\n".join(lines)


# --- Readings: what each indicator is showing, before the model sees it ------
# These describe the indicator value only. The SHAP value supplies the second
# half of each sentence, which is what ties the reading to the model's decision.


def _reading_sma_20(row: pd.Series) -> str:
    close, sma = float(row["Close"]), float(row["sma_20"])
    gap = (close / sma - 1.0) * 100 if sma else 0.0
    if gap > 0.5:
        return f"The price is {gap:.1f}% above its 20-day average."
    if gap < -0.5:
        return f"The price is {abs(gap):.1f}% below its 20-day average."
    return "The price is sitting close to its 20-day average."


def _reading_rsi_14(row: pd.Series) -> str:
    rsi = float(row["rsi_14"])
    if rsi >= 70:
        return f"RSI is {rsi:.0f}, in overbought territory after a sharp rise."
    if rsi <= 30:
        return f"RSI is {rsi:.0f}, in oversold territory after a sharp fall."
    return f"RSI is {rsi:.0f}, a normal reading \u2014 neither overbought nor oversold."


def _reading_macd(row: pd.Series) -> str:
    v = float(row["macd"])
    state = "positive" if v > 0 else "negative" if v < 0 else "flat"
    return f"The MACD line is {state}, reflecting the recent trend direction."


def _reading_macd_signal(row: pd.Series) -> str:
    v = float(row["macd_signal"])
    state = "positive" if v > 0 else "negative" if v < 0 else "flat"
    return f"The MACD signal line is {state}, a smoothed view of the same trend."


def _reading_macd_hist(row: pd.Series) -> str:
    v = float(row["macd_hist"])
    if v > 0:
        return "The MACD histogram is positive, suggesting upward momentum."
    if v < 0:
        return "The MACD histogram is negative, suggesting downward momentum."
    return "The MACD histogram is flat, with no clear momentum either way."


def _reading_bb_width(row: pd.Series) -> str:
    w = float(row["bb_width"])
    return (
        f"The Bollinger Bands are {w:.2f} wide, a measure of how volatile "
        "the price has been recently."
    )


def _reading_vol_ratio_20(row: pd.Series) -> str:
    v = float(row["vol_ratio_20"])
    if v > 1.2:
        return f"Trading volume is {v:.1f}x its recent average \u2014 unusually busy."
    if v < 0.8:
        return f"Trading volume is {v:.1f}x its recent average \u2014 unusually quiet."
    return "Trading volume is close to its recent average."


def _reading_momentum(row: pd.Series, col: str, days: str) -> str:
    pct = float(row[col]) * 100
    if pct > 0.5:
        return f"The price is up {pct:.1f}% over the last {days}."
    if pct < -0.5:
        return f"The price is down {abs(pct):.1f}% over the last {days}."
    return f"The price is broadly flat over the last {days} ({pct:+.1f}%)."


READINGS = {
    "sma_20": _reading_sma_20,
    "rsi_14": _reading_rsi_14,
    "macd": _reading_macd,
    "macd_signal": _reading_macd_signal,
    "macd_hist": _reading_macd_hist,
    "bb_width": _reading_bb_width,
    "vol_ratio_20": _reading_vol_ratio_20,
    "momentum_1d": lambda r: _reading_momentum(r, "momentum_1d", "day"),
    "momentum_5d": lambda r: _reading_momentum(r, "momentum_5d", "5 days"),
    "momentum_20d": lambda r: _reading_momentum(r, "momentum_20d", "20 days"),
}


def _push_phrase(shap_value: float, rank: int) -> str:
    """Describe how strongly and in which direction this feature moved the model."""
    direction = "toward BUY" if shap_value > 0 else "toward SELL"
    if abs(shap_value) < 1e-9:
        return "It had no measurable effect on the model's decision."
    strength = {
        0: "This was the strongest factor pushing the model",
        1: "This was the second-largest factor pushing the model",
        2: "This was the third-largest factor pushing the model",
    }.get(rank, "This pushed the model")
    return f"{strength} {direction}."


def explain_from_shap(
    shap_row: pd.Series,
    indicator_row: pd.Series,
    recommendation: int,
    n_signals: int = 3,
) -> ModelExplanation:
    """Render the model's own attributions for one prediction as plain English.

    Args:
        shap_row: Signed SHAP values for this prediction, indexed by feature
            name. Positive means the feature pushed toward BUY.
        indicator_row: The indicator values for the same day, including "Close".
        recommendation: The model's prediction (BUY or SELL constant).
        n_signals: How many of the strongest features to describe.

    Returns:
        A ModelExplanation whose signals are ordered by absolute SHAP value.

    Raises:
        ValueError: if the recommendation is not BUY or SELL.
    """
    if recommendation not in (BUY, SELL):
        raise ValueError(f"recommendation must be BUY ({BUY}) or SELL ({SELL}).")

    ranked = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)
    top = ranked.head(n_signals)

    signals: list[DrivenSignal] = []
    for rank, (feature, value) in enumerate(top.items()):
        reader = READINGS.get(feature)
        reading = reader(indicator_row) if reader else f"{feature} was a factor."
        signals.append(
            DrivenSignal(
                feature=feature,
                display_name=DISPLAY_NAMES.get(feature, feature),
                shap_value=float(value),
                reading=reading,
                push=_push_phrase(float(value), rank),
            )
        )

    rec_word = "BUY" if recommendation == BUY else "SELL"
    if recommendation == BUY:
        headline = (
            "The model leans BUY for this stock. Below is what actually drove "
            "that call, ranked by how much each factor moved the decision."
        )
    else:
        headline = (
            "The model leans SELL (or staying out) for this stock. Below is what "
            "actually drove that call, ranked by how much each factor moved the "
            "decision."
        )

    caveat = (
        "This explains how the model reached its decision. It is not evidence "
        "that the decision is correct \u2014 see the accuracy figures in the "
        "project report. This tool is for learning about market indicators, not "
        "for making investment decisions."
    )

    return ModelExplanation(
        recommendation=rec_word,
        headline=headline,
        signals=signals,
        caveat=caveat,
    )