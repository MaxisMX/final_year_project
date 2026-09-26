"""
Tests for src.models.gru.
These deliberately use a TINY lookback and few epochs so they run fast 
the goal is to verify the leakage-safe plumbing, not to train a good model:
  - The scaler is fit on training data only (the core leakage guarantee).
  - Predictions align to the correct dates (last day of each window).
  - Train and test windowing never straddle the boundary.
  - Predictions are binary.
  - Too-small slices raise clearly.

GRU training is slow and slightly non-deterministic, so we do NOT assert exact
accuracy values - only shapes, alignment, and the leakage guarantees. These
mirror tests/test_lstm.py so the two sequential models are verified identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gru import predict_gru, train_gru
from src.models.random_forest import FEATURE_COLUMNS


def _frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    data["label_5d"] = rng.integers(0, 2, n)
    return pd.DataFrame(data, index=idx)


def test_train_returns_artifacts_with_fitted_scaler() -> None:
    df = _frame(120)
    art = train_gru(df, lookback=10, epochs=2)
    assert hasattr(art.scaler, "mean_")
    assert len(art.scaler.mean_) == len(FEATURE_COLUMNS)
    assert art.lookback == 10


def test_scaler_fit_on_train_only() -> None:
    """
    The scaler's learned mean must match the TRAINING data, not test data.
    We fit on one frame, then check the scaler's mean equals that frame's mean -
    proving it never saw any other (test) data.
    """
    df = _frame(120, seed=1)
    art = train_gru(df, lookback=10, epochs=2)
    expected_means = df[FEATURE_COLUMNS].mean().to_numpy()
    np.testing.assert_allclose(art.scaler.mean_, expected_means, rtol=1e-5)


def test_predictions_align_to_last_day_of_windows() -> None:
    train = _frame(120, seed=2)
    test = _frame(60, seed=3)
    art = train_gru(train, lookback=10, epochs=2)
    preds, y_true = predict_gru(art, test)
    # With lookback=10 on 60 test rows: 60 - 10 + 1 = 51 predictions.
    assert len(preds) == 51
    assert len(y_true) == 51
    expected_first_date = test.index[10 - 1]
    assert preds.index[0] == expected_first_date
    assert preds.index[-1] == test.index[-1]


def test_predictions_are_binary() -> None:
    train = _frame(120, seed=4)
    test = _frame(50, seed=5)
    art = train_gru(train, lookback=10, epochs=2)
    preds, _ = predict_gru(art, test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_train_rejects_too_small_slice() -> None:
    tiny = _frame(5)
    with pytest.raises(ValueError, match="too small"):
        train_gru(tiny, lookback=60, epochs=2)


def test_test_slice_too_small_raises() -> None:
    train = _frame(120, seed=6)
    art = train_gru(train, lookback=10, epochs=2)
    tiny_test = _frame(5, seed=7)
    with pytest.raises(ValueError, match="too small"):
        predict_gru(art, tiny_test)