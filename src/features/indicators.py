"""Technical indicators computed from OHLCV data.

Phase 1.2 deliverable. We start with two indicators:
  - Simple Moving Average (SMA): the average closing price over a window.
  - Relative Strength Index (RSI): a momentum oscillator (0-100) measuring the
    speed and magnitude of recent price changes.

Design principles:
  - Every function takes a DataFrame (or Series) and RETURNS a new Series. It
    never mutates the input. This keeps the feature pipeline composable and
    avoids accidental in-place corruption.
  - Indicators are inherently backward-looking (they use only past and current
    data), which is what we want: no future information leaks in. The early
    rows of each indicator are NaN until enough history exists to compute them.
    We leave those NaNs in place — dropping them is a downstream decision.
  - Each indicator has a test that checks it against a hand-worked example, so
    a silent maths bug cannot slip through.
"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average of the closing price.

    Args:
        close: Series of closing prices, indexed by date.
        window: Number of periods to average over.

    Returns:
        Series of the rolling mean. The first (window - 1) values are NaN.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return close.rolling(window=window).mean().rename(f"sma_{window}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing method.

    RSI = 100 - (100 / (1 + RS)), where RS = avg gain / avg loss over `window`.

    This implementation uses Wilder's original smoothing (an exponential moving
    average with alpha = 1/window), which is the standard definition used by
    most charting platforms. A simpler rolling-mean version exists but gives
    slightly different numbers; we use Wilder's so results match common tools
    and are verifiable against published examples.

    Args:
        close: Series of closing prices, indexed by date.
        window: Lookback period (14 is the conventional default).

    Returns:
        Series of RSI values in [0, 100]. Leading values are NaN until enough
        history exists.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing must be seeded with a SIMPLE mean of the first `window`
    # deltas, then smoothed recursively: avg = (prev_avg * (window-1) + new) / window.
    # pandas' ewm does NOT seed this way, which silently skews the early values
    # (verified against Wilder's published example, expected ~70.53). So we
    # compute the recurrence explicitly.
    avg_gain = _wilder_smooth(gain, window)
    avg_loss = _wilder_smooth(loss, window)

    rs = avg_gain / avg_loss
    rsi_series = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss is 0 (only gains in the window), RS is inf and RSI -> 100.
    rsi_series = rsi_series.where(avg_loss != 0, 100.0)
    # Preserve NaNs from the warm-up period.
    rsi_series = rsi_series.where(avg_gain.notna())

    return rsi_series.rename(f"rsi_{window}")


def _wilder_smooth(values: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing: simple mean seed, then recursive smoothing.

    The first valid value (at index position `window`, since values[0] is the
    NaN from .diff()) is the simple mean of the first `window` real values.
    Each subsequent value is (prev * (window - 1) + current) / window.
    Returns a Series aligned to the input index, NaN during warm-up.
    """
    result = pd.Series(index=values.index, dtype="float64")
    vals = values.to_numpy()
    # values[0] is NaN (from diff); the first `window` real deltas are positions 1..window.
    if len(vals) <= window:
        return result  # not enough data; all NaN

    # Seed: simple mean of the first `window` real values (positions 1..window inclusive).
    seed = vals[1 : window + 1].mean()
    result.iloc[window] = seed
    prev = seed
    for i in range(window + 1, len(vals)):
        prev = (prev * (window - 1) + vals[i]) / window
        result.iloc[i] = prev
    return result


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the current indicator set to an OHLCV frame.

    Returns a NEW frame with indicator columns added. Input is not mutated.
    As we add indicators in later phases, this is the single place to register
    them so the rest of the pipeline picks them up automatically.
    """
    out = df.copy()
    out["sma_20"] = sma(df["Close"], window=20)
    out["rsi_14"] = rsi(df["Close"], window=14)
    return out


if __name__ == "__main__":
    # Manual smoke run against cached MSFT data:
    #   python -m src.features.indicators
    from src.data.fetcher import fetch_ohlcv

    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    print("Columns:", feat.columns.tolist())
    print("\nLast 5 rows (indicators populated):")
    print(feat[["Close", "sma_20", "rsi_14"]].tail())
    print("\nRSI range check — should sit within [0, 100]:")
    print(f"  min: {feat['rsi_14'].min():.2f}, max: {feat['rsi_14'].max():.2f}")