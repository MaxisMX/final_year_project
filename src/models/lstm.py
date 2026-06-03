"""LSTM classifier for BUY/SELL prediction.

Phase 2.3 (part 2) — your proposal's main model.

What's different from the Random Forest
---------------------------------------
The RF sees one day's features at a time. The LSTM sees an ordered window of the
last `lookback` days (default 60, per your proposal) and learns temporal
patterns across that window. This is the whole reason it MIGHT beat the RF — but
it also might not, and that is a legitimate finding.

Two leakage protections specific to the LSTM
--------------------------------------------
  1. SCALING: LSTMs need normalised inputs (unlike tree models). The scaler is
     fit on TRAINING data only, then applied to test data. Fitting on all data
     would leak test-set statistics into training — a classic, silent leak.
  2. WINDOWING PER SPLIT: train and test sequences are built separately so no
     window straddles the boundary (which would put training days inside a test
     window). See src/models/sequences.py.

Reproducibility
---------------
Seeds are set, but note TensorFlow's oneDNN optimisations mean LSTM results are
slightly less reproducible than the RF (small floating-point differences run to
run). Worth a sentence in the report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Quieten TF's startup logging before import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402

from src.models.random_forest import FEATURE_COLUMNS, EvalResult  # noqa: E402
from src.models.sequences import make_sequences  # noqa: E402

DEFAULT_LOOKBACK = 60
DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for as much reproducibility as TF allows."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm(lookback: int, n_features: int, seed: int = DEFAULT_SEED) -> Sequential:
    """Build a compact LSTM classifier.

    Deliberately small (one LSTM layer + dropout) to limit overfitting on noisy
    financial data — a large network will memorise noise and generalise worse.
    Dropout further regularises. Output is a single sigmoid (BUY probability).
    """
    set_seeds(seed)
    model = Sequential(
        [
            Input(shape=(lookback, n_features)),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@dataclass
class LSTMArtifacts:
    """A trained LSTM plus the scaler fit on its training data."""

    model: Sequential
    scaler: StandardScaler
    lookback: int
    feature_columns: list[str]


def train_lstm(
    train_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "label_5d",
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> LSTMArtifacts:
    """Train an LSTM on a training slice (leakage-safe scaling + windowing).

    The scaler is fit ONLY on this training slice. EarlyStopping halts training
    when validation loss stops improving, using the last 20% of the (time-ordered)
    training data as a validation tail — itself a mini chronological split, so
    no shuffling.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    set_seeds(seed)

    feats = train_df[feature_columns]
    labels = train_df[label_column].astype(int)

    # Fit scaler on training features ONLY.
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(feats), index=feats.index, columns=feature_columns
    )

    X, y = make_sequences(scaled, labels, lookback=lookback)
    if len(X) == 0:
        raise ValueError(
            f"Training slice too small for lookback={lookback}: "
            f"{len(train_df)} rows yields no windows."
        )

    model = build_lstm(lookback, len(feature_columns), seed=seed)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    # validation_split uses the LAST fraction of X (time-ordered) — Keras does
    # not shuffle before splitting when validation_split is set, so this stays
    # chronological. We also pass shuffle=False to be explicit.
    model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
        shuffle=False,
    )
    return LSTMArtifacts(
        model=model,
        scaler=scaler,
        lookback=lookback,
        feature_columns=list(feature_columns),
    )


def predict_lstm(
    artifacts: LSTMArtifacts,
    test_df: pd.DataFrame,
    label_column: str = "label_5d",
    threshold: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Predict BUY/SELL on a test slice using a trained LSTM.

    The test features are scaled with the TRAINING scaler (not refit), and
    windowed separately so no window straddles the train/test boundary.

    Returns:
        (predictions, true_labels) as aligned Series. Because each prediction
        needs `lookback` days of history, the first (lookback - 1) test rows
        cannot be predicted; both Series are aligned to the predictable dates.
    """
    feats = test_df[artifacts.feature_columns]
    labels = test_df[label_column].astype(int)

    scaled = pd.DataFrame(
        artifacts.scaler.transform(feats), index=feats.index, columns=artifacts.feature_columns
    )
    X, y = make_sequences(scaled, labels, lookback=artifacts.lookback)
    if len(X) == 0:
        raise ValueError(
            f"Test slice too small for lookback={artifacts.lookback}."
        )

    probs = artifacts.model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    # Each window's prediction corresponds to its LAST day. Align to those dates.
    pred_dates = test_df.index[artifacts.lookback - 1 :]
    return (
        pd.Series(preds, index=pred_dates, name="prediction"),
        pd.Series(y, index=pred_dates, name=label_column),
    )


def evaluate_lstm(
    artifacts: LSTMArtifacts, test_df: pd.DataFrame, label_column: str = "label_5d"
) -> EvalResult:
    """Evaluate an LSTM on a test slice, reusing the RF's honest metric set."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from src.features.target import BUY, SELL

    preds, y_true = predict_lstm(artifacts, test_df, label_column=label_column)

    majority_class = y_true.mode().iloc[0]
    majority_acc = accuracy_score(y_true, [majority_class] * len(y_true))

    cm = confusion_matrix(y_true, preds, labels=[SELL, BUY])
    cm_df = pd.DataFrame(
        cm, index=["actual_SELL", "actual_BUY"], columns=["pred_SELL", "pred_BUY"]
    )
    return EvalResult(
        accuracy=accuracy_score(y_true, preds),
        precision=precision_score(y_true, preds, pos_label=BUY, zero_division=0),
        recall=recall_score(y_true, preds, pos_label=BUY, zero_division=0),
        f1=f1_score(y_true, preds, pos_label=BUY, zero_division=0),
        majority_baseline_accuracy=majority_acc,
        confusion=cm_df,
        report=classification_report(
            y_true, preds, target_names=["SELL", "BUY"], zero_division=0
        ),
    )


if __name__ == "__main__":
    # Single train/test LSTM run against cached MSFT data:
    #   python -m src.models.lstm
    from src.data.fetcher import fetch_ohlcv
    from src.features.indicators import add_indicators
    from src.features.target import attach_label, drop_unlabelled
    from src.models.random_forest import chronological_split

    print("Building features + labels...")
    data = fetch_ohlcv("MSFT")
    feat = add_indicators(data)
    labelled = attach_label(feat, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

    split = chronological_split(clean)
    train_df = clean.loc[: split.split_date].iloc[:-1]  # train portion
    test_df = clean.loc[split.split_date :]

    print(f"Training LSTM on {len(train_df)} rows (lookback=60)... this is slower than RF.")
    artifacts = train_lstm(train_df, lookback=60, verbose=1)

    print("\nEvaluating on test set...")
    result = evaluate_lstm(artifacts, test_df)
    print(f"Accuracy:          {result.accuracy:.3f}")
    print(f"Majority baseline: {result.majority_baseline_accuracy:.3f}")
    print(f"Beats baseline?    {'YES' if result.beats_majority() else 'NO'}")
    print(f"F1 (BUY):          {result.f1:.3f}")
    print("\nConfusion matrix:")
    print(result.confusion)
    print("\nReminder: the LSTM may NOT beat the RF or the baseline. That is a")
    print("legitimate, reportable finding — do not force it to look better.")