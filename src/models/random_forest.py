"""Random Forest baseline classifier for BUY/SELL prediction.

Phase 1.4 deliverable. This closes the thin end-to-end slice:
    data -> features -> label -> train/test split -> model -> evaluation.

What this is (and is NOT)
-------------------------
This is a BASELINE, deliberately simple. Its two jobs:
  1. Prove the whole pipeline runs end-to-end.
  2. Produce an honest number for the LSTM to beat later.

It is NOT meant to be impressive. On 5-day stock direction, accuracy in the
low-to-mid 50s% is normal and expected. Markets are close to random at this
horizon. If you ever see 70%+ accuracy, treat it as a LEAKAGE BUG until proven
otherwise, not a success.

Evaluation discipline
----------------------
  - CHRONOLOGICAL split only: train on earlier dates, test on later dates.
    Never shuffle time-series — that leaks the future into training.
  - We report accuracy AND precision/recall/F1, because the classes are
    usually imbalanced (an up-trending stock has more BUY days). Accuracy alone
    would flatter a model that just always predicts the majority class, so we
    also compare against that 'always predict majority' baseline explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.target import BUY, SELL

# Feature columns the model trains on. As we add indicators in Phase 2, register
# them here so the model picks them up. Keep this explicit — never just "all
# columns", or you risk accidentally feeding the label or a future-derived
# column into the model (leakage).
FEATURE_COLUMNS = [
    "sma_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "vol_ratio_20",
    "momentum_1d",
    "momentum_5d",
    "momentum_20d",
]


@dataclass
class SplitData:
    """Holds a chronological train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    split_date: pd.Timestamp


@dataclass
class EvalResult:
    """Evaluation metrics for a trained model on the test set."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    majority_baseline_accuracy: float
    confusion: pd.DataFrame
    report: str

    def beats_majority(self) -> bool:
        """Did the model actually beat 'always predict the majority class'?"""
        return self.accuracy > self.majority_baseline_accuracy


def chronological_split(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "label_5d",
    train_fraction: float = 0.8,
) -> SplitData:
    """Split a labelled feature frame chronologically (no shuffling).

    The earliest `train_fraction` of rows become the training set; the rest are
    the test set. Rows must already be in date order and fully labelled (call
    drop_unlabelled first) and have no NaN features (the indicator warm-up rows
    should be dropped before calling this).

    Raises:
        ValueError: if required columns are missing or NaNs remain.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    missing = [c for c in [*feature_columns, label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Features contain NaN values. Drop indicator warm-up rows before "
            "splitting (e.g. df.dropna(subset=feature_columns))."
        )
    if df[label_column].isna().any():
        raise ValueError(
            "Label contains NaN values. Call drop_unlabelled() before splitting."
        )

    df = df.sort_index()
    cut = int(len(df) * train_fraction)
    if cut == 0 or cut == len(df):
        raise ValueError(f"train_fraction={train_fraction} leaves an empty split.")

    train, test = df.iloc[:cut], df.iloc[cut:]
    return SplitData(
        X_train=train[feature_columns],
        X_test=test[feature_columns],
        y_train=train[label_column].astype(int),
        y_test=test[label_column].astype(int),
        split_date=test.index[0],
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int | None = 5,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    max_depth is capped (5) by default to limit overfitting on noisy financial
    features — a deep forest will happily memorise noise. random_state is fixed
    so results are reproducible (important for the report).
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",  # counteract class imbalance
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> EvalResult:
    """Evaluate a trained model honestly, including the majority-class baseline."""
    y_pred = model.predict(X_test)

    # The 'always predict the majority class' baseline — the bar a useful model
    # must clear. If accuracy barely beats this, the model has learned little.
    majority_class = y_test.mode().iloc[0]
    majority_acc = accuracy_score(y_test, [majority_class] * len(y_test))

    cm = confusion_matrix(y_test, y_pred, labels=[SELL, BUY])
    cm_df = pd.DataFrame(
        cm,
        index=["actual_SELL", "actual_BUY"],
        columns=["pred_SELL", "pred_BUY"],
    )

    return EvalResult(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, pos_label=BUY, zero_division=0),
        recall=recall_score(y_test, y_pred, pos_label=BUY, zero_division=0),
        f1=f1_score(y_test, y_pred, pos_label=BUY, zero_division=0),
        majority_baseline_accuracy=majority_acc,
        confusion=cm_df,
        report=classification_report(
            y_test, y_pred, target_names=["SELL", "BUY"], zero_division=0
        ),
    )


if __name__ == "__main__":
    # Full thin-slice run against cached MSFT data:
    #   python -m src.models.random_forest
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled

    print("Loading data and building features + label...")
    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)

    # Drop indicator warm-up NaNs AND the unlabelled tail.
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)
    print(f"Usable rows after dropping warm-up + unlabelled tail: {len(clean)}")

    split = chronological_split(clean)
    print(f"Train: {len(split.X_train)} rows | Test: {len(split.X_test)} rows")
    print(f"Test period starts: {split.split_date.date()}")

    print("\nTraining Random Forest...")
    model = train_random_forest(split.X_train, split.y_train)
    result = evaluate(model, split.X_test, split.y_test)

    print("\n" + "=" * 50)
    print("RESULTS (test set)")
    print("=" * 50)
    print(f"Accuracy:           {result.accuracy:.3f}")
    print(f"Majority baseline:  {result.majority_baseline_accuracy:.3f}")
    print(f"Beats baseline?     {'YES' if result.beats_majority() else 'NO'}")
    print(f"Precision (BUY):    {result.precision:.3f}")
    print(f"Recall (BUY):       {result.recall:.3f}")
    print(f"F1 (BUY):           {result.f1:.3f}")
    print("\nConfusion matrix:")
    print(result.confusion)
    print("\nFull report:")
    print(result.report)
    print("Reminder: low-to-mid 50s%% accuracy is NORMAL here. 70%%+ => suspect leakage.")  