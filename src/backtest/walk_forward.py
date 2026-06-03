"""Walk-forward validation for time-series-honest model evaluation.

Phase 2.2 deliverable. Replaces the single chronological split with a sequence
of expanding-window folds, giving a DISTRIBUTION of performance rather than one
number.

Why this matters
----------------
A single train/test split tells you how the model did on ONE test period. That
could be luck — an unusually trending or choppy stretch. Walk-forward trains and
tests repeatedly across different periods, so you can report e.g. "accuracy
0.52 +/- 0.03 across 6 folds" and know whether the model is CONSISTENTLY okay
or just got lucky once. This is the clearest signal of time-series evaluation
maturity you can show, and it directly mirrors how a strategy would be re-trained
and used over time.

Expanding window
----------------
Each fold trains on ALL data up to a point, then tests on the next chunk:

    Fold 1: train[0:n]            test[n:2n]
    Fold 2: train[0:2n]          test[2n:3n]
    Fold 3: train[0:3n]          test[3n:4n]
    ...

Training data only ever precedes test data (no shuffling, no future leakage),
and the training set grows each fold so it always uses all available history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.random_forest import FEATURE_COLUMNS, evaluate, train_random_forest


@dataclass
class FoldResult:
    """Metrics for one walk-forward fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    accuracy: float
    precision: float
    recall: float
    f1: float
    majority_baseline_accuracy: float


@dataclass
class WalkForwardResult:
    """Aggregated results across all folds."""

    folds: list[FoldResult]

    def _column(self, attr: str) -> np.ndarray:
        return np.array([getattr(f, attr) for f in self.folds])

    def mean_accuracy(self) -> float:
        return float(self._column("accuracy").mean())

    def std_accuracy(self) -> float:
        return float(self._column("accuracy").std())

    def mean_f1(self) -> float:
        return float(self._column("f1").mean())

    def mean_majority_baseline(self) -> float:
        return float(self._column("majority_baseline_accuracy").mean())

    def beats_baseline_on_average(self) -> bool:
        return self.mean_accuracy() > self.mean_majority_baseline()

    def summary_frame(self) -> pd.DataFrame:
        """Per-fold metrics as a DataFrame for reporting / plotting."""
        return pd.DataFrame(
            [
                {
                    "fold": f.fold,
                    "test_start": f.test_start.date(),
                    "test_end": f.test_end.date(),
                    "accuracy": round(f.accuracy, 4),
                    "f1": round(f.f1, 4),
                    "majority_baseline": round(f.majority_baseline_accuracy, 4),
                }
                for f in self.folds
            ]
        )


def walk_forward_validate(
    df: pd.DataFrame,
    n_splits: int = 5,
    feature_columns: list[str] | None = None,
    label_column: str = "label_5d",
    min_train_size: int | None = None,
) -> WalkForwardResult:
    """Run expanding-window walk-forward validation.

    The data is divided into (n_splits + 1) equal time blocks. Fold i trains on
    blocks [0..i] and tests on block i+1, so every test block is preceded by all
    earlier data. The first block is the initial training set (never used as a
    test set), guaranteeing each fold has a non-trivial training history.

    Args:
        df: Labelled, NaN-free feature frame in date order (drop warm-up rows
            and unlabelled tail before calling).
        n_splits: Number of train/test folds.
        feature_columns: Columns to use as features (defaults to FEATURE_COLUMNS).
        label_column: Name of the label column.
        min_train_size: If set, skip folds whose training set is smaller than this.

    Returns:
        WalkForwardResult with per-fold metrics and aggregate statistics.

    Raises:
        ValueError: on missing columns, NaNs, or too little data for n_splits.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    missing = [c for c in [*feature_columns, label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[feature_columns].isna().any().any():
        raise ValueError("Features contain NaN. Drop warm-up rows before validating.")
    if df[label_column].isna().any():
        raise ValueError("Label contains NaN. Call drop_unlabelled() before validating.")

    df = df.sort_index()
    n = len(df)
    if n < (n_splits + 1) * 2:
        raise ValueError(
            f"Not enough data ({n} rows) for {n_splits} splits. "
            f"Need at least {(n_splits + 1) * 2} rows."
        )

    # (n_splits + 1) equal blocks; block boundaries via integer division.
    block = n // (n_splits + 1)
    folds: list[FoldResult] = []

    for i in range(n_splits):
        train_end_idx = block * (i + 1)
        test_end_idx = block * (i + 2) if i < n_splits - 1 else n

        train = df.iloc[:train_end_idx]
        test = df.iloc[train_end_idx:test_end_idx]

        if min_train_size is not None and len(train) < min_train_size:
            continue
        if len(test) == 0:
            continue

        X_train = train[feature_columns]
        y_train = train[label_column].astype(int)
        X_test = test[feature_columns]
        y_test = test[label_column].astype(int)

        # A fold needs both classes present in training to be meaningful.
        if y_train.nunique() < 2:
            continue

        model = train_random_forest(X_train, y_train)
        ev = evaluate(model, X_test, y_test)

        folds.append(
            FoldResult(
                fold=i + 1,
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                accuracy=ev.accuracy,
                precision=ev.precision,
                recall=ev.recall,
                f1=ev.f1,
                majority_baseline_accuracy=ev.majority_baseline_accuracy,
            )
        )

    if not folds:
        raise ValueError("No valid folds produced. Check data size and class balance.")

    return WalkForwardResult(folds=folds)


if __name__ == "__main__":
    # Walk-forward run against cached MSFT data:
    #   python -m src.backtest.walk_forward
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled

    print("Building features + labels, running walk-forward validation...")
    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    result = walk_forward_validate(clean, n_splits=5)

    print("\nPer-fold results:")
    print(result.summary_frame().to_string(index=False))

    print("\n" + "=" * 50)
    print("AGGREGATE (across all folds)")
    print("=" * 50)
    print(f"Mean accuracy:      {result.mean_accuracy():.3f} "
          f"(+/- {result.std_accuracy():.3f})")
    print(f"Mean majority base: {result.mean_majority_baseline():.3f}")
    print(f"Mean F1 (BUY):      {result.mean_f1():.3f}")
    print(f"Beats baseline on average? "
          f"{'YES' if result.beats_baseline_on_average() else 'NO'}")
    print(
        "\nThe std tells the real story: a small std means consistent performance;\n"
        "a large std means the single-split number was partly luck. Either way,\n"
        "report the mean +/- std, not one number."
    )