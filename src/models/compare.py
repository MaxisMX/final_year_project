"""
Fair RF vs LSTM vs GRU vs CNN-LSTM comparison under identical walk-forward folds.

Phase 2.5 deliverable, extended in response to supervisor feedback that Random
Forest and LSTM alone are weak predictors. This compares four architectures
across three model families:

    - Random Forest  (tree-based, non-sequential)
    - LSTM           (recurrent, three gates)
    - GRU            (recurrent, two gates - simpler than LSTM)
    - CNN-LSTM       (convolutional feature extraction + recurrent)

Questions answered
------------------
    1. Does any sequential model beat the non-sequential Random Forest?
    2. Does the LSTM's extra gating buy anything over the simpler GRU?
    3. Does convolutional feature extraction (CNN-LSTM) help over a plain LSTM?
    4. Does ANY model beat a naive majority-class baseline?

How fairness is enforced
------------------------
  - SAME folds, SAME data, SAME metrics, SAME baseline.
  - SAME scoring dates: all sequence models share a lookback, so they reach the
    same dates; RF is restricted to those same dates. RF is not judged on early
    days the sequence models cannot see.
  - SAME capacity: LSTM, GRU and the CNN-LSTM's recurrent layer all use 32
    units, and the two deep models share an epoch budget, so differences are
    attributable to architecture rather than size or training time.

Why confusion matrices matter here
----------------------------------
Accuracy alone is misleading. A model can match the baseline by predicting one
class for every day. The per-fold confusion matrices make this visible: a
column of zeros (or near-zeros) means the model collapsed to a constant guess.
Read the matrices alongside the accuracy before claiming any model has skill.

Honesty note
------------
All models may sit at or below the baseline, and different models may collapse
toward DIFFERENT classes. That divergence is itself a finding: it indicates the
features carry no generalisable directional signal, so each model defaults to
whatever prior its training setup nudges it toward. This module reports what the
folds show; it does not tune anything to look better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.features.target import BUY, SELL
from src.models.cnn_lstm import predict_cnn_lstm, train_cnn_lstm
from src.models.gru import predict_gru, train_gru
from src.models.lstm import predict_lstm, train_lstm
from src.models.random_forest import FEATURE_COLUMNS, train_random_forest

# Model keys in report order.
MODEL_KEYS = ("rf", "lstm", "gru", "cnn_lstm")
MODEL_LABELS = {
    "rf": "RandomForest",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
}


def _confusion_df(y_true: pd.Series, preds: pd.Series) -> pd.DataFrame:
    """
    Confusion matrix as a labelled frame, matching EvalResult's convention.
    """
    cm = confusion_matrix(y_true, preds, labels=[SELL, BUY])
    return pd.DataFrame(
        cm, index=["actual_SELL", "actual_BUY"], columns=["pred_SELL", "pred_BUY"]
    )


def _collapse_direction(conf: pd.DataFrame) -> str | None:
    """Return 'SELL'/'BUY' if the model predicted (almost) only that class.

    Uses a 5% threshold rather than exact zero, since a model that predicts the
    minority class a handful of times is functionally collapsed. Returns None if
    the model made a genuine two-sided prediction.
    """
    col_totals = conf.sum(axis=0)
    total = col_totals.sum()
    if total == 0:
        return None
    sell_frac = col_totals["pred_SELL"] / total
    buy_frac = col_totals["pred_BUY"] / total
    if buy_frac <= 0.05:
        return "SELL"
    if sell_frac <= 0.05:
        return "BUY"
    return None


@dataclass
class ModelFoldScore:
    """One model's score on one fold, on the common (sequence-reachable) dates."""

    fold: int
    accuracy: float
    f1: float
    confusion: pd.DataFrame

    def collapse_direction(self) -> str | None:
        """'SELL'/'BUY' if the model collapsed to that class, else None."""
        return _collapse_direction(self.confusion)

    def collapsed(self) -> bool:
        return self.collapse_direction() is not None


@dataclass
class ComparisonFold:
    """All models' scores on a single fold, plus the shared baseline."""

    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_common_dates: int
    rf: ModelFoldScore
    lstm: ModelFoldScore
    gru: ModelFoldScore
    cnn_lstm: ModelFoldScore
    majority_baseline_accuracy: float


@dataclass
class ComparisonResult:
    """Aggregated comparison across all folds."""

    folds: list[ComparisonFold]

    def rf_mean_accuracy(self) -> float:
        return self.mean_accuracy("rf")

    def lstm_mean_accuracy(self) -> float:
        return self.mean_accuracy("lstm")

    def gru_mean_accuracy(self) -> float:
        return self.mean_accuracy("gru")

    def cnn_lstm_mean_accuracy(self) -> float:
        return self.mean_accuracy("cnn_lstm")

    def _arr(self, model: str, metric: str) -> np.ndarray:
        return np.array([getattr(getattr(f, model), metric) for f in self.folds])

    def mean_accuracy(self, model: str) -> float:
        return float(self._arr(model, "accuracy").mean())

    def std_accuracy(self, model: str) -> float:
        return float(self._arr(model, "accuracy").std())

    def mean_f1(self, model: str) -> float:
        return float(self._arr(model, "f1").mean())

    def mean_baseline(self) -> float:
        return float(
            np.array([f.majority_baseline_accuracy for f in self.folds]).mean()
        )

    def winner(self, tolerance: float = 0.005) -> str:
        means = {k: self.mean_accuracy(k) for k in MODEL_KEYS}
        ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        (best_key, best_val), (_, second_val) = ranked[0], ranked[1]
        if best_val - second_val < tolerance:
            return "tie"
        return MODEL_LABELS[best_key]

    def beats_baseline(self, model: str, tolerance: float = 0.005) -> bool:
        return self.mean_accuracy(model) - self.mean_baseline() > tolerance

    def any_beats_baseline(self, tolerance: float = 0.005) -> bool:
        return any(self.beats_baseline(k, tolerance) for k in MODEL_KEYS)

    def collapse_counts(self) -> dict[str, int]:
        """How many folds each model collapsed to a single class on."""
        return {
            key: sum(getattr(f, key).collapsed() for f in self.folds)
            for key in MODEL_KEYS
        }

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
                    "gru_acc": round(f.gru.accuracy, 4),
                    "cnn_acc": round(f.cnn_lstm.accuracy, 4),
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
    gru_epochs: int | None = None,
    cnn_epochs: int | None = None,
) -> ComparisonResult:
    """
    Run RF, LSTM, GRU and CNN-LSTM through identical expanding-window folds.
    Args:
        gru_epochs, cnn_epochs: Epoch budgets for the deep models. Both default
            to `lstm_epochs` so all three deep models get an identical budget -
            important for a fair comparison.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    gru_epochs = lstm_epochs if gru_epochs is None else gru_epochs
    cnn_epochs = lstm_epochs if cnn_epochs is None else cnn_epochs

    missing = [c for c in [*feature_columns, label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[feature_columns].isna().any().any() or df[label_column].isna().any():
        raise ValueError("NaNs present. Drop warm-up rows and unlabelled tail first.")

    df = df.sort_index()
    n = len(df)
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

        # Plain LSTM: defines the common scoring dates 
        lstm_art = train_lstm(
            train_df,
            feature_columns=feature_columns,
            label_column=label_column,
            lookback=lookback,
            epochs=lstm_epochs,
        )
        lstm_preds, y_common = predict_lstm(
            lstm_art, test_df, label_column=label_column
        )
        common_dates = lstm_preds.index

        #  GRU: same lookback, so same reachable dates 
        gru_art = train_gru(
            train_df,
            feature_columns=feature_columns,
            label_column=label_column,
            lookback=lookback,
            epochs=gru_epochs,
        )
        gru_preds, _ = predict_gru(gru_art, test_df, label_column=label_column)

        # --- CNN-LSTM: same lookback, so same reachable dates ---
        cnn_art = train_cnn_lstm(
            train_df,
            feature_columns=feature_columns,
            label_column=label_column,
            lookback=lookback,
            epochs=cnn_epochs,
        )
        cnn_preds, _ = predict_cnn_lstm(cnn_art, test_df, label_column=label_column)

        # All three sequence models share a lookback, so their scoring dates
        # must be identical. Assert rather than assume - a silent misalignment
        # would quietly invalidate the comparison.
        for name, preds in (("GRU", gru_preds), ("CNN-LSTM", cnn_preds)):
            if not preds.index.equals(common_dates):
                raise ValueError(
                    f"Fold {i + 1}: {name} scoring dates do not match the LSTM's. "
                    f"This should be impossible with a shared lookback and "
                    f"indicates a windowing bug."
                )

        # RF: train on same data, predict, then RESTRICT to common dates 
        rf_model = train_random_forest(train_df[feature_columns], y_train)
        rf_pred_all = pd.Series(
            rf_model.predict(test_df[feature_columns]), index=test_df.index
        )
        rf_preds = rf_pred_all.loc[common_dates]

        majority_class = y_common.mode().iloc[0]
        baseline_acc = accuracy_score(y_common, [majority_class] * len(y_common))

        def _score(preds: pd.Series) -> ModelFoldScore:
            return ModelFoldScore(
                fold=i + 1,
                accuracy=accuracy_score(y_common, preds),
                f1=f1_score(y_common, preds, pos_label=BUY, zero_division=0),
                confusion=_confusion_df(y_common, preds),
            )

        folds.append(
            ComparisonFold(
                fold=i + 1,
                test_start=common_dates[0],
                test_end=common_dates[-1],
                n_common_dates=len(common_dates),
                rf=_score(rf_preds),
                lstm=_score(lstm_preds),
                gru=_score(gru_preds),
                cnn_lstm=_score(cnn_preds),
                majority_baseline_accuracy=baseline_acc,
            )
        )

    if not folds:
        raise ValueError("No valid folds produced. Check data size, lookback, balance.")

    return ComparisonResult(folds=folds)


if __name__ == "__main__":
    # Full fair comparison against cached MSFT data.
    # SLOW - trains THREE neural networks per fold:
    #   python -m src.models.compare
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled

    data_name = "TSLA"
    print("Building features + labels...")
    data = fetch_ohlcv(data_name)
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    print(
        "Running fair walk-forward comparison "
        "(trains LSTM + GRU + CNN-LSTM per fold - be patient)..."
    )
    result = compare_models(clean, n_splits=5, lookback=60)

    print("\nPer-fold accuracy (scored on identical dates):")
    print(result.summary_frame().to_string(index=False))

    # Per-fold confusion matrices - the key diagnostic.
    print("\n" + "=" * 62)
    print("PER-FOLD CONFUSION MATRICES")
    print("=" * 62)
    for f in result.folds:
        print(f"\n--- Fold {f.fold}  ({f.test_start.date()} to {f.test_end.date()}) ---")
        for key in MODEL_KEYS:
            score = getattr(f, key)
            direction = score.collapse_direction()
            flag = f"   << COLLAPSED to {direction}" if direction else ""
            print(f"\n{MODEL_LABELS[key]} (acc {score.accuracy:.3f}){flag}")
            print(score.confusion.to_string())

    print("\n" + "=" * 62)
    print("AGGREGATE - fair four-way comparison")
    print("=" * 62)
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        print(
            f"{label:<14} {result.mean_accuracy(key):.3f} "
            f"(+/- {result.std_accuracy(key):.3f})"
        )
    print(f"{'Majority base':<14} {result.mean_baseline():.3f}")

    collapse = result.collapse_counts()
    print(f"\nFolds collapsed to a single class (out of {len(result.folds)}):")
    for key in MODEL_KEYS:
        print(f"  {MODEL_LABELS[key]:<14} {collapse[key]}")

    print(f"\nHighest mean accuracy: {result.winner()}")
    print(f"Any model beats baseline? "
          f"{'YES' if result.any_beats_baseline() else 'NO'}")
    print(
        "\nInterpretation guidance:\n"
        "  - Read the confusion matrices FIRST. A COLLAPSED model did not learn;\n"
        "    its accuracy just tracks the base rate of whichever class it picked.\n"
        "  - If different models collapse toward DIFFERENT classes, that is\n"
        "    evidence the features carry no generalisable directional signal -\n"
        "    each model defaults to its own prior rather than a learned pattern.\n"
        "  - Compare gaps against the fold-to-fold std before claiming a winner.\n"
        "  - If no model beats the baseline, say so plainly. That is a real\n"
        "    finding, consistent with the published literature."
    )
    print(f"Fetched data: {data_name}")