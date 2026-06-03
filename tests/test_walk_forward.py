"""Tests for src.backtest.walk_forward.

Key guarantees:
  - Each fold's training data ends strictly before its test data begins
    (the walk-forward leakage guard).
  - The training window EXPANDS fold over fold.
  - Aggregate statistics (mean/std) are computed correctly.
  - Guard rails fire on bad input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import walk_forward_validate
from src.models.random_forest import FEATURE_COLUMNS


def _make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    # Balanced-ish labels so every fold has both classes.
    data["label_5d"] = rng.integers(0, 2, n)
    return pd.DataFrame(data, index=dates)


def test_each_fold_trains_before_it_tests() -> None:
    """The core leakage guard: train_end must precede test_start in every fold."""
    result = walk_forward_validate(_make_frame(), n_splits=5)
    for fold in result.folds:
        assert fold.train_end < fold.test_start, (
            f"Fold {fold.fold}: training data ({fold.train_end}) overlaps or "
            f"follows test data ({fold.test_start}) — leakage!"
        )


def test_training_window_expands() -> None:
    """Expanding window: each fold's training span should be >= the previous."""
    result = walk_forward_validate(_make_frame(), n_splits=5)
    train_spans = [(f.train_end - f.train_start).days for f in result.folds]
    for earlier, later in zip(train_spans, train_spans[1:], strict=False):
        assert later >= earlier


def test_produces_requested_number_of_folds() -> None:
    result = walk_forward_validate(_make_frame(n=600), n_splits=5)
    assert len(result.folds) == 5
    assert [f.fold for f in result.folds] == [1, 2, 3, 4, 5]


def test_aggregate_statistics_are_consistent() -> None:
    result = walk_forward_validate(_make_frame(), n_splits=4)
    accs = [f.accuracy for f in result.folds]
    assert result.mean_accuracy() == pytest.approx(np.mean(accs))
    assert result.std_accuracy() == pytest.approx(np.std(accs))


def test_metrics_in_valid_range() -> None:
    result = walk_forward_validate(_make_frame(), n_splits=5)
    for f in result.folds:
        for metric in (f.accuracy, f.precision, f.recall, f.f1):
            assert 0.0 <= metric <= 1.0


def test_random_data_does_not_beat_baseline_much() -> None:
    """On pure-noise features, mean accuracy must stay near chance (leakage guard)."""
    result = walk_forward_validate(_make_frame(seed=99), n_splits=5)
    assert result.mean_accuracy() < 0.65, (
        "Walk-forward accuracy too high on random data — suspect leakage."
    )


def test_summary_frame_shape() -> None:
    result = walk_forward_validate(_make_frame(), n_splits=5)
    sf = result.summary_frame()
    assert len(sf) == 5
    assert {"fold", "test_start", "test_end", "accuracy", "f1"}.issubset(sf.columns)


def test_rejects_missing_columns() -> None:
    df = _make_frame().drop(columns=["macd"])
    with pytest.raises(ValueError, match="Missing required columns"):
        walk_forward_validate(df)


def test_rejects_nan_features() -> None:
    df = _make_frame()
    df.loc[df.index[0], "rsi_14"] = np.nan
    with pytest.raises(ValueError, match="Features contain NaN"):
        walk_forward_validate(df)


def test_rejects_insufficient_data() -> None:
    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward_validate(_make_frame(n=8), n_splits=5)