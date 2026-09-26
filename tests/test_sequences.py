"""
Tests for src.models.sequences.

The windowing is the most leakage-prone part of the LSTM, so these tests verify
the shape maths and  critically  that each window's label is the label of the
LAST day in that window (not the first, a common off-by-one bug).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.sequences import make_sequences


def _frame(n: int, n_features: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.bdate_range("2020-01-01", periods=n)
    feats = pd.DataFrame(
        {f"f{j}": np.arange(n) * 10 + j for j in range(n_features)},
        index=idx,
        dtype=float,
    )
    labels = pd.Series(np.arange(n) % 2, index=idx)  # 0,1,0,1,...
    return feats, labels


def test_output_shapes() -> None:
    feats, labels = _frame(100, n_features=3)
    X, y = make_sequences(feats, labels, lookback=60)
    # n_windows = 100 - 60 + 1 = 41
    assert X.shape == (41, 60, 3)
    assert y.shape == (41,)


def test_label_is_last_day_of_window() -> None:
    """Window i must be paired with the label of its LAST row (i+lookback-1)."""
    feats, labels = _frame(10, n_features=2)
    X, y = make_sequences(feats, labels, lookback=3)
    # Window 0 covers rows 0,1,2 -> label should be labels[2]
    assert y[0] == labels.iloc[2]
    # Window 1 covers rows 1,2,3 -> label should be labels[3]
    assert y[1] == labels.iloc[3]
    # Last window covers rows 7,8,9 -> label should be labels[9]
    assert y[-1] == labels.iloc[9]


def test_window_contents_are_correct_rows() -> None:
    """The features in window i must be exactly rows [i : i+lookback]."""
    feats, labels = _frame(10, n_features=2)
    X, _ = make_sequences(feats, labels, lookback=3)
    # Window 0, first row, should equal feature row 0.
    np.testing.assert_array_almost_equal(X[0, 0], feats.iloc[0].to_numpy())
    # Window 0, last row, should equal feature row 2.
    np.testing.assert_array_almost_equal(X[0, 2], feats.iloc[2].to_numpy())
    # Window 1, first row, should equal feature row 1.
    np.testing.assert_array_almost_equal(X[1, 0], feats.iloc[1].to_numpy())


def test_too_few_rows_returns_empty() -> None:
    feats, labels = _frame(5, n_features=3)
    X, y = make_sequences(feats, labels, lookback=60)
    assert X.shape == (0, 60, 3)
    assert y.shape == (0,)


def test_lookback_one_is_one_row_per_window() -> None:
    feats, labels = _frame(5, n_features=2)
    X, y = make_sequences(feats, labels, lookback=1)
    assert X.shape == (5, 1, 2)
    # Each label is just that day's label.
    np.testing.assert_array_equal(y, labels.to_numpy())


def test_rejects_length_mismatch() -> None:
    feats, labels = _frame(10)
    with pytest.raises(ValueError, match="length mismatch"):
        make_sequences(feats, labels.iloc[:-1], lookback=3)


def test_rejects_bad_lookback() -> None:
    feats, labels = _frame(10)
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        make_sequences(feats, labels, lookback=0)