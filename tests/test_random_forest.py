"""Tests for src.models.random_forest.

The key guarantees we test:
  - The split is CHRONOLOGICAL (train strictly before test, no shuffling). This
    is the leakage guard for this module.
  - Guard rails fire on NaN features/labels and missing columns.
  - The model trains and the evaluation produces sane, in-range metrics, plus
    an honest majority-class baseline to compare against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.random_forest import (
    chronological_split,
    evaluate,
    train_random_forest,
)


def _make_labelled_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic labelled feature frame with a date index.

    Generates every column in FEATURE_COLUMNS so the model's input contract is
    satisfied as the indicator set grows.
    """
    from src.models.random_forest import FEATURE_COLUMNS

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    data["label_5d"] = rng.integers(0, 2, n)
    return pd.DataFrame(data, index=dates)


def test_split_is_chronological() -> None:
    """Every training date must be strictly before every test date."""
    df = _make_labelled_frame()
    split = chronological_split(df, train_fraction=0.8)
    assert split.X_train.index.max() < split.X_test.index.min()


def test_split_sizes_respect_fraction() -> None:
    df = _make_labelled_frame(n=100)
    split = chronological_split(df, train_fraction=0.8)
    assert len(split.X_train) == 80
    assert len(split.X_test) == 20


def test_split_rejects_nan_features() -> None:
    df = _make_labelled_frame()
    df.loc[df.index[0], "sma_20"] = np.nan
    with pytest.raises(ValueError, match="Features contain NaN"):
        chronological_split(df)


def test_split_rejects_nan_labels() -> None:
    df = _make_labelled_frame()
    df["label_5d"] = df["label_5d"].astype("float")
    df.loc[df.index[0], "label_5d"] = np.nan
    with pytest.raises(ValueError, match="Label contains NaN"):
        chronological_split(df)


def test_split_rejects_missing_columns() -> None:
    df = _make_labelled_frame().drop(columns=["rsi_14"])
    with pytest.raises(ValueError, match="Missing required columns"):
        chronological_split(df)


def test_split_rejects_degenerate_fraction() -> None:
    df = _make_labelled_frame(n=10)
    with pytest.raises(ValueError, match="empty split"):
        chronological_split(df, train_fraction=1.0)


def test_train_and_evaluate_produces_valid_metrics() -> None:
    df = _make_labelled_frame(n=300)
    split = chronological_split(df)
    model = train_random_forest(split.X_train, split.y_train, n_estimators=50)
    result = evaluate(model, split.X_test, split.y_test)

    # All metrics must be valid proportions.
    for value in (result.accuracy, result.precision, result.recall, result.f1):
        assert 0.0 <= value <= 1.0
    assert 0.0 <= result.majority_baseline_accuracy <= 1.0

    # Confusion matrix totals must equal the test set size.
    assert result.confusion.to_numpy().sum() == len(split.y_test)


def test_evaluate_reports_majority_baseline() -> None:
    """On pure-random labels, the model should not meaningfully beat the baseline.

    This is a sanity check: with noise features and random labels, there is no
    signal to learn, so a leakage-free pipeline must NOT produce high accuracy.
    """
    df = _make_labelled_frame(n=400, seed=7)
    split = chronological_split(df)
    model = train_random_forest(split.X_train, split.y_train, n_estimators=50)
    result = evaluate(model, split.X_test, split.y_test)
    # Random data => accuracy should be near the baseline, definitely not >0.7.
    assert result.accuracy < 0.7, (
        "Suspiciously high accuracy on random data suggests a leakage bug."
    )