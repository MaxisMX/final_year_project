"""
SHAP feature-importance analysis for the Random Forest model.

Phase 3 (part 2). Where the plain-English generator interprets the INDICATORS,
SHAP interprets the MODEL: it quantifies how much each feature pushed a given
prediction toward BUY or SELL. This is genuine model interpretability.

Why SHAP on the Random Forest and not on LSTM
--------------------------------------------
SHAP's TreeExplainer is exact and fast for tree models. On sequential LSTMs,
SHAP is awkward and slow (and conceptually messy across 60 timesteps), so per
the project plan we run rigorous SHAP on the RF and treat LSTM interpretability
as a separate, smaller question. State this scoping choice in the report it's
a deliberate, defensible decision, not an omission.

Output shape note
-----------------
For a binary classifier this SHAP version returns an array of shape
(n_samples, n_features, n_classes). We take the BUY-class slice (index 1) so a
POSITIVE SHAP value means "pushed the prediction toward BUY" and negative means
"pushed toward SELL". This makes the values directly interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from src.features.target import BUY


@dataclass
class ShapAnalysis:
    """Holds SHAP values and convenience summaries for a set of predictions."""

    shap_values: pd.DataFrame  # rows = samples, cols = features, signed (toward BUY)
    feature_names: list[str]

    def global_importance(self) -> pd.Series:
        """
        Mean absolute SHAP value per feature — overall importance ranking.

        Higher = the feature mattered more to the model's decisions overall,
        regardless of direction. This is the headline 'which indicators drive
        the model?' result.
        """
        return (
            self.shap_values.abs().mean().sort_values(ascending=False)
        )

    def top_features_for_row(self, row_index: int, n: int = 3) -> pd.Series:
        """
        The n features with the largest (signed) push for one prediction.

        Positive = pushed toward BUY, negative = pushed toward SELL. Useful for
        explaining a single recommendation: 'RSI and MACD drove this call'.
        """
        row = self.shap_values.iloc[row_index]
        return row.reindex(row.abs().sort_values(ascending=False).index).head(n)


def compute_shap(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> ShapAnalysis:
    """
    Compute SHAP values for a trained Random Forest on samples X.

    Args:
        model: A fitted RandomForestClassifier.
        X: Feature rows to explain (the same columns the model was trained on).

    Returns:
        ShapAnalysis with signed SHAP values oriented toward the BUY class.
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    arr = np.array(raw)
    # Binary classifier -> shape (n_samples, n_features, n_classes). Take BUY class.
    if arr.ndim == 3:
        # Class axis is last; BUY is the positive class (label 1).
        buy_class_idx = list(model.classes_).index(BUY)
        values = arr[:, :, buy_class_idx]
    elif arr.ndim == 2:
        # Some versions return (n_samples, n_features) already for the positive class.
        values = arr
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

    shap_df = pd.DataFrame(values, columns=list(X.columns), index=X.index)
    return ShapAnalysis(shap_values=shap_df, feature_names=list(X.columns))


if __name__ == "__main__":
    # SHAP analysis against cached MSFT data:
    #   python -m src.explain.shap_analysis
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled
    from src.models.random_forest import (
        FEATURE_COLUMNS,
        chronological_split,
        train_random_forest,
    )

    print("Building features, training RF, computing SHAP...")
    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    split = chronological_split(clean)
    model = train_random_forest(split.X_train, split.y_train)

    # Explain the test set (or a sample of it for speed).
    sample = split.X_test.iloc[:200]
    analysis = compute_shap(model, sample)

    print("\n" + "=" * 50)
    print("GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")
    print("=" * 50)
    importance = analysis.global_importance()
    for feat_name, val in importance.items():
        bar = "#" * int(val * 200)
        print(f"  {feat_name:16} {val:.4f} {bar}")

    print("\n" + "=" * 50)
    print("EXPLAINING ONE PREDICTION (most recent test row)")
    print("=" * 50)
    top = analysis.top_features_for_row(len(sample) - 1, n=3)
    for feat_name, val in top.items():
        direction = "toward BUY" if val > 0 else "toward SELL"
        print(f"  {feat_name:16} {val:+.4f}  ({direction})")

    print(
        "\nInterpretation: these show what drove the MODEL's decisions, which"
        "\ncomplements the plain-English generator's reading of the indicators."
        "\nNote: high importance does NOT mean the model is accurate \u2014 it only"
        "\nshows which features the (near-baseline) model relied on."
    )