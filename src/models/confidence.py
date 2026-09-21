"""Detecting when a model's prediction should not be trusted or explained.

Phase 5 — the second half of the explainability fix.

Why this module exists
----------------------
Section 5.4 of the report found that several models collapse: they predict the
same class for every day, scoring close to the majority-class baseline while
learning nothing. A collapsed model still produces a confident-looking verdict,
and SHAP will still attribute that verdict to features. Rendering those
attributions as fluent sentences would manufacture a rationale for what is
really a default — the system would sound most convincing exactly where it is
least informative.

So before any explanation is generated, the service asks this module whether
the model is behaving like a predictor at all. Three outcomes:

    COLLAPSED       one class for every recent day. No explanation is shown;
                    the user is told no directional signal is available.
    LOW_CONFIDENCE  the model discriminates, but this particular prediction sits
                    near the decision boundary. The explanation is shown with a
                    warning attached.
    OK              the model varies its output and this prediction is clear of
                    the boundary.

Thresholds
----------
Both thresholds are conventional choices rather than tuned values, and are set
here so they are documented and easy to defend or change:

  MIN_MINORITY_SHARE = 0.05
      If fewer than 5% of recent predictions are the minority class, the model
      is treated as collapsed. A strict "exactly one class" test would miss a
      model that predicts BUY on 119 of 120 days, which is collapse in all but
      name.

  MIN_MARGIN = 0.05
      A predicted probability within 0.05 of 0.5 means the model is barely
      separating the classes for this input. Flagged, not suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features.target import BUY

MIN_MINORITY_SHARE = 0.05
MIN_MARGIN = 0.05
DEFAULT_WINDOW = 120


class Confidence(str, Enum):
    """How much weight the system should place on a prediction."""

    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    COLLAPSED = "collapsed"


@dataclass
class ConfidenceReport:
    """The result of checking whether a prediction is worth explaining."""

    status: Confidence
    buy_probability: float  # model's probability for the BUY class, this day
    margin: float  # distance of that probability from 0.5
    minority_share: float  # share of the rarer class over the recent window
    window_days: int
    message: str

    @property
    def should_explain(self) -> bool:
        """Whether an explanation should be generated at all."""
        return self.status is not Confidence.COLLAPSED

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "buy_probability": round(self.buy_probability, 4),
            "margin": round(self.margin, 4),
            "minority_share": round(self.minority_share, 4),
            "window_days": self.window_days,
            "message": self.message,
        }


def _buy_column(model: RandomForestClassifier) -> int:
    """Index of the BUY class within predict_proba's output columns."""
    return list(model.classes_).index(BUY)


def assess_confidence(
    model: RandomForestClassifier,
    X_recent: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> ConfidenceReport:
    """Decide whether this model's latest prediction should be trusted.

    The latest prediction is the final row of X_recent. The rows before it are
    used only to judge whether the model varies its output at all.

    Args:
        model: A fitted RandomForestClassifier.
        X_recent: Feature rows in date order, ending with the day to predict.
            At least a few dozen rows are needed for the collapse check to mean
            anything.
        window: How many recent days to inspect for collapse.

    Returns:
        A ConfidenceReport. Callers should check `should_explain` before
        generating any explanation text.

    Raises:
        ValueError: if X_recent is empty.
    """
    if len(X_recent) == 0:
        raise ValueError("X_recent must contain at least one row.")

    recent = X_recent.iloc[-window:]
    buy_col = _buy_column(model)

    preds = pd.Series(model.predict(recent), index=recent.index)
    counts = preds.value_counts()
    minority_share = float(counts.min() / len(preds)) if len(counts) > 1 else 0.0

    buy_probability = float(model.predict_proba(recent.iloc[[-1]])[0][buy_col])
    margin = abs(buy_probability - 0.5)

    if minority_share < MIN_MINORITY_SHARE:
        only_class = "BUY" if preds.iloc[-1] == BUY else "SELL"
        return ConfidenceReport(
            status=Confidence.COLLAPSED,
            buy_probability=buy_probability,
            margin=margin,
            minority_share=minority_share,
            window_days=len(recent),
            message=(
                f"No reliable signal is available for this stock. Over the last "
                f"{len(recent)} trading days the model has predicted {only_class} "
                f"almost every day, regardless of what the indicators were doing. "
                f"That is a default rather than a judgement, so no explanation is "
                f"offered here \u2014 there is no reasoning behind it to describe."
            ),
        )

    if margin < MIN_MARGIN:
        return ConfidenceReport(
            status=Confidence.LOW_CONFIDENCE,
            buy_probability=buy_probability,
            margin=margin,
            minority_share=minority_share,
            window_days=len(recent),
            message=(
                "The model is close to undecided on this stock today: its "
                f"probability for BUY is {buy_probability:.0%}, near the 50% "
                "coin-flip point. Treat the explanation below as a description "
                "of a marginal call, not a clear signal."
            ),
        )

    return ConfidenceReport(
        status=Confidence.OK,
        buy_probability=buy_probability,
        margin=margin,
        minority_share=minority_share,
        window_days=len(recent),
        message=(
            f"The model varied its predictions over the last {len(recent)} "
            "trading days, so this call reflects the indicators rather than a "
            "fixed default."
        ),
    )