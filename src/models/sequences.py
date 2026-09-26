"""
Sequence windowing for the LSTM model.

Phase 2.3 (part 1). The LSTM does NOT see one day at a time like the Random
Forest. It sees an ordered WINDOW of the last `lookback` days, and predicts the
label for the final day of that window. This module turns the flat feature
table into those windows — and it is the single most leakage-prone piece of the
LSTM pipeline, so it lives in its own module with its own tests.

The shape transformation
------------------------
Input:  a feature frame of shape (n_days, n_features) and a label per day.
Output: X of shape (n_windows, lookback, n_features)
        y of shape (n_windows,)

Window i covers feature rows [i : i + lookback] and is paired with the label at
row (i + lookback - 1) — the label of the LAST day in the window. So the model
learns: "given the last `lookback` days of indicators, what is the label today?"

Leakage safety
--------------
  - Windows are built ONLY from rows you pass in. To keep train and test windows
    from straddling the boundary, you must window the train slice and the test
    slice SEPARATELY (the model module does this). Never window the whole frame
    and then split — that lets test windows contain training days.
  - The features inside a window are all <= the day being labelled, and the
    label is the forward-return label already shown (in target.py) to be
    leakage-safe. No future information enters the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a feature frame + labels into LSTM sequence windows.

    Args:
        features: DataFrame of shape (n_days, n_features), date-ordered, NaN-free.
        labels: Series of labels aligned to `features` (same index, same length).
        lookback: Number of days per window.

    Returns:
        (X, y) where X has shape (n_windows, lookback, n_features) and y has
        shape (n_windows,). n_windows = len(features) - lookback + 1.
        If there are fewer rows than `lookback`, returns empty arrays.

    Raises:
        ValueError: if features and labels are misaligned or lookback < 1.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if len(features) != len(labels):
        raise ValueError(
            f"features ({len(features)}) and labels ({len(labels)}) length mismatch."
        )
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must share the same index.")

    n_features = features.shape[1]
    feat_values = features.to_numpy(dtype="float32")
    label_values = labels.to_numpy()

    n_windows = len(features) - lookback + 1
    if n_windows <= 0:
        return (
            np.empty((0, lookback, n_features), dtype="float32"),
            np.empty((0,), dtype="int64"),
        )

    X = np.empty((n_windows, lookback, n_features), dtype="float32")
    y = np.empty((n_windows,), dtype="int64")
    for i in range(n_windows):
        X[i] = feat_values[i : i + lookback]
        # Label of the LAST day in the window.
        y[i] = label_values[i + lookback - 1]
    return X, y