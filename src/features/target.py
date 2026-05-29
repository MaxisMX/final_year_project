"""Define the prediction target (label) for the BUY/SELL classifier.

Phase 1.3 deliverable. This is the most leakage-prone part of the project, so
the discipline is explicit and tested.

The target
----------
For each day t, we look at the closing price `horizon` trading days into the
future. If that future close is higher than today's close, the label is BUY (1),
otherwise SELL (0).

    forward_return(t) = close(t + horizon) / close(t) - 1
    label(t) = 1 (BUY) if forward_return(t) > 0 else 0 (SELL)

Why this is leakage-safe
------------------------
  - FEATURES (computed elsewhere) use only data up to and including day t.
  - The LABEL looks only at day t + horizon, strictly in the future.
  - These two windows never overlap, so no future information leaks into the
    inputs the model sees at prediction time.
  - The final `horizon` rows have NO label (the future doesn't exist yet in the
    data). They are returned as NaN and MUST be dropped before training. Failing
    to drop them, or accidentally filling them, is a classic silent bug — so we
    make the NaNs explicit and test for them.

A note on honesty
-----------------
This label ignores transaction costs and assumes you can transact at the close.
That's standard for a first pass, but the backtest (Phase 1.5+) is where we
confront whether the strategy survives real-world frictions. Don't let a good
label-prediction accuracy fool you into thinking the strategy is profitable.
"""

from __future__ import annotations

import pandas as pd

# Label encoding — keep these named so the rest of the project never hard-codes 0/1.
BUY = 1
SELL = 0


def forward_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Return over the next `horizon` trading days.

    forward_return(t) = close(t + horizon) / close(t) - 1

    The last `horizon` values are NaN because their future is not in the data.

    Args:
        close: Series of closing prices, indexed by date.
        horizon: Number of trading days to look ahead.

    Returns:
        Series of forward returns, aligned to day t. Last `horizon` rows are NaN.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    future_close = close.shift(-horizon)
    return (future_close / close - 1.0).rename(f"fwd_return_{horizon}")


def make_label(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Binary BUY/SELL label from the sign of the forward return.

    BUY (1) if the price `horizon` days ahead is higher than today, else SELL (0).
    The last `horizon` rows are NaN (no future available) and must be dropped
    before training.

    Args:
        close: Series of closing prices, indexed by date.
        horizon: Look-ahead window in trading days.

    Returns:
        Series of {0, 1} with NaN in the final `horizon` rows.
    """
    fwd = forward_return(close, horizon)
    # Where forward return is known, label by its sign; where NaN, stay NaN.
    label = pd.Series(pd.NA, index=close.index, dtype="Int64", name=f"label_{horizon}d")
    label[fwd > 0] = BUY
    label[fwd <= 0] = SELL
    # Re-impose NaN on the warm-down rows (fwd is NaN there).
    label[fwd.isna()] = pd.NA
    return label


def attach_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Attach the label column to a feature frame, without mutating the input.

    Returns a NEW frame. The caller is responsible for dropping rows where the
    label is NaN before training (use `drop_unlabelled`).
    """
    out = df.copy()
    out[f"label_{horizon}d"] = make_label(df["Close"], horizon)
    return out


def drop_unlabelled(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Drop rows where the label is NaN (the final `horizon` rows).

    This is the step people forget. Keeping unlabelled rows in the training set
    will either crash the model or, worse, get them silently filled somewhere.
    """
    col = f"label_{horizon}d"
    if col not in df.columns:
        raise KeyError(f"No label column '{col}' found. Call attach_label first.")
    return df.dropna(subset=[col]).copy()


if __name__ == "__main__":
    # Manual smoke run against cached MSFT data:
    #   python -m src.features.target
    from src.data.fetcher import fetch_ohlcv

    data = fetch_ohlcv("MSFT")
    labelled = attach_label(data, horizon=5)
    print(f"Total rows: {len(labelled)}")
    print(f"Rows with NaN label (should equal horizon=5): {labelled['label_5d'].isna().sum()}")

    clean = drop_unlabelled(labelled, horizon=5)
    counts = clean["label_5d"].value_counts()
    buys = counts.get(BUY, 0)
    sells = counts.get(SELL, 0)
    total = buys + sells
    print(f"\nAfter dropping unlabelled rows: {len(clean)}")
    print(f"  BUY:  {buys} ({100 * buys / total:.1f}%)")
    print(f"  SELL: {sells} ({100 * sells / total:.1f}%)")
    print("\nClass balance matters: if this is wildly skewed (e.g. 70/30), note it —")
    print("it affects which metrics are meaningful (accuracy alone would mislead).")