"""Fair RF-vs-LSTM comparison under identical walk-forward folds.

Phase 2.5 deliverable — the centrepiece of the modelling section. This answers
the proposal's central claim: does the LSTM actually outperform the Random
Forest on this problem?

How fairness is enforced
------------------------
  - SAME folds: both models use identical expanding-window train/test splits.
  - SAME data: identical features and labels.
  - SAME metrics: accuracy, F1, vs the same majority baseline.
  - SAME scoring dates within each fold: the LSTM needs `lookback` days of
    history before its first prediction, so it cannot score the first
    (lookback - 1) days of a test slice. To compare like-for-like, we evaluate
    BOTH models only on the dates the LSTM can reach. Without this alignment the
    RF would be judged on extra early days the LSTM never sees — an unfair edge.

Honesty note
------------
Both models may sit below the majority baseline (earlier runs suggested so).
That is a legitimate, reportable finding. This module does not try to make
either model look better than it is; it reports what the folds show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.features.target import BUY
from src.models.lstm import predict_lstm, train_lstm
from src.models.random_forest import FEATURE_COLUMNS, train_random_forest


@dataclass
class ModelFoldScore:
    """One model's score on one fold, on the common (LSTM-reachable) dates."""

    fold: int
    accuracy: float
    f1: float


@dataclass
class ComparisonFold:
    """Both models' scores on a single fold, plus the shared baseline."""

    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_common_dates: int
    rf: ModelFoldScore
    lstm: ModelFoldScore
    majority_baseline_accuracy: float


@dataclass
class ComparisonResult:
    """Aggregated RF-vs-LSTM comparison across all folds."""

    folds: list[ComparisonFold]

    def _arr(self, model: str, metric: str) -> np.ndarray:
        return np.array([getattr(getattr(f, model), metric) for f in self.folds])

    def rf_mean_accuracy(self) -> float:
        return float(self._arr("rf", "accuracy").mean())

    def rf_std_accuracy(self) -> float:
        return float(self._arr("rf", "accuracy").std())

    def lstm_mean_accuracy(self) -> float:
        return float(self._arr("lstm", "accuracy").mean())

    def lstm_std_accuracy(self) -> float:
        return float(self._arr("lstm", "accuracy").std())

    def mean_baseline(self) -> float:
        return float(np.array([f.majority_baseline_accuracy for f in self.folds]).mean())

    def winner(self) -> str:
        """Which model has the higher mean accuracy — or 'tie' if within 0.005."""
        diff = self.lstm_mean_accuracy() - self.rf_mean_accuracy()
        if abs(diff) < 0.005:
            return "tie"
        return "LSTM" if diff > 0 else "RandomForest"

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": f.fold,
                    "test_start": f.test_start.date(),
                    "test_end": f.test_end.date(),
                    "n_dates": f.n_common_dates,
                    "rf_acc": round(f.rf.accuracy, 4),
                    "lstm_acc": round(f.lstm.accuracy, 4),
                    "baseline": round(f.majority_baseline_accuracy, 4),
                }
                for f in self.folds
            ]
        )


def compare_models(
    df: pd.DataFrame,
    n_splits: int = 5,
    lookback: int = 60,
    feature_columns: list[str] | None = None,
    label_column: str = "label_5d",
    lstm_epochs: int = 30,
) -> ComparisonResult:
    """Run RF and LSTM through identical expanding-window walk-forward folds.

    Each fold trains both models on the same training block and scores both on
    the SAME dates of the test block (the dates the LSTM can reach, given its
    lookback). Returns per-fold and aggregate comparison statistics.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    missing = [c for c in [*feature_columns, label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[feature_columns].isna().any().any() or df[label_column].isna().any():
        raise ValueError("NaNs present. Drop warm-up rows and unlabelled tail first.")

    df = df.sort_index()
    n = len(df)
    # Each fold's test slice must be larger than lookback or the LSTM gets no
    # predictable dates. Require generous room.
    if n < (n_splits + 1) * (lookback + 5):
        raise ValueError(
            f"Not enough data ({n} rows) for {n_splits} folds with lookback "
            f"{lookback}. Need ~{(n_splits + 1) * (lookback + 5)} rows."
        )

    block = n // (n_splits + 1)
    folds: list[ComparisonFold] = []

    for i in range(n_splits):
        train_end_idx = block * (i + 1)
        test_end_idx = block * (i + 2) if i < n_splits - 1 else n

        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx:test_end_idx]

        y_train = train_df[label_column].astype(int)
        if y_train.nunique() < 2:
            continue
        if len(test_df) <= lookback:
            continue

        # --- LSTM: train and predict (defines the common scoring dates) ---
        lstm_art = train_lstm(
            train_df,
            feature_columns=feature_columns,
            label_column=label_column,
            lookback=lookback,
            epochs=lstm_epochs,
        )
        lstm_preds, y_common = predict_lstm(lstm_art, test_df, label_column=label_column)
        # The LSTM can only score from (lookback-1) onward; these are the common dates.
        common_dates = lstm_preds.index

        # --- RF: train on same data, predict, then RESTRICT to common dates ---
        rf_model = train_random_forest(train_df[feature_columns], y_train)
        rf_pred_all = pd.Series(
            rf_model.predict(test_df[feature_columns]), index=test_df.index
        )
        rf_preds = rf_pred_all.loc[common_dates]

        # Baseline on the common dates only (fair shared reference).
        majority_class = y_common.mode().iloc[0]
        baseline_acc = accuracy_score(y_common, [majority_class] * len(y_common))

        folds.append(
            ComparisonFold(
                fold=i + 1,
                test_start=common_dates[0],
                test_end=common_dates[-1],
                n_common_dates=len(common_dates),
                rf=ModelFoldScore(
                    fold=i + 1,
                    accuracy=accuracy_score(y_common, rf_preds),
                    f1=f1_score(y_common, rf_preds, pos_label=BUY, zero_division=0),
                ),
                lstm=ModelFoldScore(
                    fold=i + 1,
                    accuracy=accuracy_score(y_common, lstm_preds),
                    f1=f1_score(y_common, lstm_preds, pos_label=BUY, zero_division=0),
                ),
                majority_baseline_accuracy=baseline_acc,
            )
        )

    if not folds:
        raise ValueError("No valid folds produced. Check data size, lookback, balance.")

    return ComparisonResult(folds=folds)


if __name__ == "__main__":
    # Full fair comparison against cached MSFT data (SLOW — trains an LSTM per fold):
    #   python -m src.models.compare
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled

    print("Building features + labels...")
    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    print("Running fair walk-forward comparison (trains one LSTM per fold — be patient)...")
    result = compare_models(clean, n_splits=5, lookback=60)

    print("\nPer-fold (scored on identical dates):")
    print(result.summary_frame().to_string(index=False))

    print("\n" + "=" * 56)
    print("AGGREGATE — fair RF vs LSTM comparison")
    print("=" * 56)
    print(f"Random Forest:  {result.rf_mean_accuracy():.3f} (+/- {result.rf_std_accuracy():.3f})")
    print(f"LSTM:           {result.lstm_mean_accuracy():.3f} (+/- {result.lstm_std_accuracy():.3f})")
    print(f"Majority base:  {result.mean_baseline():.3f}")
    print(f"Higher mean accuracy: {result.winner()}")
    print(
        "\nInterpretation guidance:\n"
        "  - If neither beats the baseline, say so plainly — it's a real finding.\n"
        "  - A small gap between RF and LSTM, relative to their std, may not be\n"
        "    meaningful. Compare the difference against the fold-to-fold spread\n"
        "    before claiming one model 'wins'."
    )