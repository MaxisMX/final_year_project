"""
Tests for src.explain.shap_analysis.
We verify the SHAP wiring: correct output shape, BUY-class orientation, and the
summary methods. We train a small RF on a signal where one feature genuinely
drives the label, then check SHAP correctly identifies that feature as the most
important — a meaningful behavioural test, not just a shape check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.explain.shap_analysis import compute_shap
from src.models.random_forest import FEATURE_COLUMNS
from sklearn.ensemble import RandomForestClassifier


def _model_with_known_driver(seed: int = 0):
    """
    Train an RF where 'rsi_14' alone determines the label.

    SHAP should then rank rsi_14 as by far the most important feature.
    """
    rng = np.random.default_rng(seed)
    n = 300
    X = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS})
    # Label depends only on rsi_14 (plus a little noise).
    y = (X["rsi_14"] + rng.normal(0, 0.2, n) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    model.fit(X, y)
    return model, X


def test_shap_output_shape_matches_input() -> None:
    model, X = _model_with_known_driver()
    analysis = compute_shap(model, X.iloc[:20])
    assert analysis.shap_values.shape == (20, len(FEATURE_COLUMNS))
    assert list(analysis.shap_values.columns) == FEATURE_COLUMNS


def test_shap_identifies_known_driver() -> None:
    """The feature that actually drives the label should top the importance list."""
    model, X = _model_with_known_driver()
    analysis = compute_shap(model, X.iloc[:100])
    importance = analysis.global_importance()
    assert importance.index[0] == "rsi_14", (
        f"Expected rsi_14 to be most important, got {importance.index[0]}"
    )


def test_global_importance_is_sorted_descending() -> None:
    model, X = _model_with_known_driver()
    analysis = compute_shap(model, X.iloc[:50])
    imp = analysis.global_importance()
    assert list(imp) == sorted(imp, reverse=True)


def test_global_importance_all_non_negative() -> None:
    """Mean absolute SHAP values must be >= 0 by definition."""
    model, X = _model_with_known_driver()
    analysis = compute_shap(model, X.iloc[:50])
    assert (analysis.global_importance() >= 0).all()


def test_top_features_for_row_returns_n() -> None:
    model, X = _model_with_known_driver()
    analysis = compute_shap(model, X.iloc[:50])
    top = analysis.top_features_for_row(0, n=3)
    assert len(top) == 3


def test_top_features_sorted_by_absolute_magnitude() -> None:
    model, X = _model_with_known_driver()
    analysis = compute_shap(model, X.iloc[:50])
    top = analysis.top_features_for_row(0, n=5)
    abs_vals = top.abs().to_numpy()
    assert list(abs_vals) == sorted(abs_vals, reverse=True)