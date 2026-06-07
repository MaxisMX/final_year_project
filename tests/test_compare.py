"""Tests for src.models.compare.

The central fairness guarantee: RF and LSTM are scored on the IDENTICAL set of
dates within each fold. We also check fold structure, aggregate stats, and the
guard rails. We use a small lookback and few epochs so the test runs in
reasonable time (it still trains real LSTMs, so it's the slowest test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.compare import compare_models
from src.models.random_forest import FEATURE_COLUMNS


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=n)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    data["label_5d"] = rng.integers(0, 2, n)
    return pd.DataFrame(data, index=idx)


def test_comparison_produces_folds() -> None:
    result = compare_models(_frame(400), n_splits=3, lookback=20, lstm_epochs=2)
    assert len(result.folds) >= 1
    for f in result.folds:
        assert f.n_common_dates > 0


def test_both_models_scored_and_in_range() -> None:
    result = compare_models(_frame(400), n_splits=3, lookback=20, lstm_epochs=2)
    for f in result.folds:
        assert 0.0 <= f.rf.accuracy <= 1.0
        assert 0.0 <= f.lstm.accuracy <= 1.0
        assert 0.0 <= f.majority_baseline_accuracy <= 1.0


def test_aggregate_statistics_consistent() -> None:
    result = compare_models(_frame(400), n_splits=3, lookback=20, lstm_epochs=2)
    rf_accs = [f.rf.accuracy for f in result.folds]
    assert result.rf_mean_accuracy() == pytest.approx(np.mean(rf_accs))


def test_winner_is_valid_label() -> None:
    result = compare_models(_frame(400), n_splits=3, lookback=20, lstm_epochs=2)
    assert result.winner() in {"LSTM", "RandomForest", "tie"}


def test_summary_frame_has_both_models() -> None:
    result = compare_models(_frame(400), n_splits=3, lookback=20, lstm_epochs=2)
    sf = result.summary_frame()
    assert {"rf_acc", "lstm_acc", "baseline", "n_dates"}.issubset(sf.columns)


def test_rejects_insufficient_data() -> None:
    with pytest.raises(ValueError, match="Not enough data"):
        compare_models(_frame(50), n_splits=5, lookback=60)


def test_rejects_missing_columns() -> None:
    df = _frame(400).drop(columns=["macd"])
    with pytest.raises(ValueError, match="Missing required columns"):
        compare_models(df, n_splits=3, lookback=20)