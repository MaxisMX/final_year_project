"""
Fetch and cache OHLCV stock data from Yahoo Finance.

Phase 1.1 deliverable. Responsibilities:
  - Download daily OHLCV data for a single ticker via yfinance.
  - Cache to local parquet so repeated runs don't hit the API.
  - Normalise yfinance's quirky output (MultiIndex columns) into a clean frame.
  - Report (but do not silently fill) missing trading days.

Design notes:
  - We use auto_adjust=True so OHLC are split/dividend adjusted. This is the
    right default for modelling: it removes artificial price jumps on dividend
    and split dates that would otherwise look like real moves.
  - We deliberately do NOT forward-fill gaps here. Filling is a modelling
    decision that belongs downstream, where it can be done knowingly. Silently
    filling at the data layer is a classic source of subtle leakage and bugs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Canonical column set we expose to the rest of the project.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Default cache location (project_root/data/raw).
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _normalise_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten yfinance output into single-level OHLCV columns.

    Modern yfinance returns MultiIndex columns like ('Close', 'MSFT') even for a
    single ticker. We flatten to just 'Close', drop any extra columns (e.g.
    'Adj Close'), and enforce a consistent column order.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level (the field name), dropping the ticker level.
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Keep only the canonical OHLCV columns, in a fixed order.
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Downloaded data for {ticker} is missing expected columns: {missing}. "
            f"Got columns: {df.columns.tolist()}"
        )
    df = df[OHLCV_COLUMNS]
    df.index.name = "Date"
    # Drop any inferred frequency on the index. Real market data is irregular
    # (holidays), so it never has a freq; clearing it here keeps freshly fetched
    # and cache-reloaded frames byte-for-byte identical.
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.freq = None
    return df


def count_missing_trading_days(df: pd.DataFrame) -> int:
    """
    Count gaps in the business-day index (excluding weekends).

    This is a sanity signal, not a fix. A handful of gaps is normal (public
    holidays). A large number suggests a data problem worth investigating.
    Note: this counts weekday gaps, so US market holidays will show up here too
    it is an upper bound on 'suspicious' missingness, not an exact figure.
    """
    if df.empty:
        return 0
    full_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
    return len(full_range.difference(df.index))


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker.upper()}_ohlcv.parquet"

def _is_stale(path: Path) -> bool:
    #True if the cached file doesn't include the most recent trading day.
    try: 
        cached = pd.read_parquet(path)
    except Exception:
        return True
    if cached.empty:
        return True
    last_cached = cached.index.max().date()
    today = pd.Timestamp.today().date() 
    last_weekday = pd.bdate_range(end=today, periods=1)[0].date()  # last trading day
    return last_cached < last_weekday

def fetch_ohlcv(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a single ticker, using a local cache.
    Args:
        ticker: Stock symbol, e.g. "MSFT".
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD), or None for "up to today".
        cache_dir: Where to store/read the parquet cache. Defaults to data/raw.
        force_refresh: If True, ignore any cached file and re-download.

    Returns:
        DataFrame indexed by Date with columns Open, High, Low, Close, Volume.

    Raises:
        ValueError: If the download returns no data (bad ticker, no network,
            or an invalid date range).
    """
    ticker = ticker.upper()
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, cache_dir)

    if path.exists() and not force_refresh and not _is_stale(path):
        logger.info("Loading %s from cache: %s", ticker, path)
        return pd.read_parquet(path)

    logger.info("Downloading %s from Yahoo Finance (%s to %s)", ticker, start, end or "today")
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the symbol, the date "
            f"range, and your internet connection."
        )

    df = _normalise_columns(raw, ticker)

    gaps = count_missing_trading_days(df)
    if gaps > 0:
        logger.info(
            "%s has %d missing weekday(s) in range (holidays + any real gaps).",
            ticker,
            gaps,
        )

    df.to_parquet(path)
    logger.info("Cached %d rows for %s to %s", len(df), ticker, path)
    return df


if __name__ == "__main__":
    # Manual smoke run: python -m src.data.fetcher
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = fetch_ohlcv("MSFT", start="2015-01-01", force_refresh=True)
    print(f"\nFetched {len(data)} rows for MSFT")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Missing weekdays in range: {count_missing_trading_days(data)}")
    print("\nFirst rows:")
    print(data.head())
    print("\nLast rows:")
    print(data.tail())
    print("\nSummary:")
    print(data.describe())