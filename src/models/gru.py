"""GRU classifier for BUY/SELL prediction.

Added alongside the LSTM and CNN-LSTM to widen the comparison across
sequential architectures, in response to supervisor feedback. The question this
module answers:

    Does a GRU - a simpler recurrent unit than the LSTM - do any better or
    worse than the LSTM on this problem?

Why GRU is a meaningful comparison
----------------------------------
LSTM and GRU are both recurrent units designed for sequential data, but the GRU
is simpler: it uses two gates (reset, update) instead of the LSTM's three
(input, forget, output), and has no separate cell state. Fewer parameters means
it trains faster and can generalise better on small datasets, where an LSTM may
overfit. On noisy daily equity data with only a few thousand rows, "simpler" is
not obviously worse - which is exactly what this comparison tests.

To keep the comparison fair, the GRU layer uses the SAME 32 units as the LSTM
in lstm.py. Any difference in results is therefore attributable to the gating
mechanism, not to a change in capacity.

Leakage protections (identical to lstm.py, deliberately)
--------------------------------------------------------
  1. SCALING: the StandardScaler is fit on TRAINING data only.
  2. WINDOWING PER SPLIT: train and test sequences are built separately so no
     window straddles the train/test boundary.

Reproducibility
---------------
Seeds are set, but TensorFlow's oneDNN optimisations mean results vary slightly
run to run - same caveat as the LSTM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import GRU, Dense, Dropout, Input  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402

from src.models.random_forest import FEATURE_COLUMNS, EvalResult  # noqa: E402
from src.models.sequences import make_sequences  # noqa: E402

DEFAULT_LOOKBACK = 60
DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for as much reproducibility as TF allows."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_gru(
    lookback: int,
    n_features: int,
    seed: int = DEFAULT_SEED,
) -> Sequential:
    """Build a compact GRU classifier.

    Architecture mirrors the LSTM exactly apart from the recurrent unit:
        GRU(32)     -> recurrent layer, same size as the LSTM for fairness
        Dropout     -> regularisation
        Dense(16)   -> hidden layer
        Dense(1)    -> sigmoid classification head (BUY probability)
    """
    set_seeds(seed)
    model = Sequential(
        [
            Input(shape=(lookback, n_features)),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@dataclass
class GRUArtifacts:
    """A trained GRU plus the scaler fit on its training data."""

    model: Sequential
    scaler: StandardScaler
    lookback: int
    feature_columns: list[str]


def train_gru(
    train_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "label_5d",
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> GRUArtifacts:
    """Train a GRU on a training slice (leakage-safe scaling + windowing).

    Mirrors train_lstm exactly apart from the model builder, so the two are
    directly comparable.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    set_seeds(seed)

    feats = train_df[feature_columns]
    labels = train_df[label_column].astype(int)

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

    model = build_gru(lookback, len(feature_columns), seed=seed)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
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
    return GRUArtifacts(
        model=model,
        scaler=scaler,
        lookback=lookback,
        feature_columns=list(feature_columns),
    )


def predict_gru(
    artifacts: GRUArtifacts,
    test_df: pd.DataFrame,
    label_column: str = "label_5d",
    threshold: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Predict BUY/SELL on a test slice using a trained GRU.

    Test features are scaled with the TRAINING scaler (not refit), and windowed
    separately so no window straddles the train/test boundary.

    Returns:
        (predictions, true_labels) as aligned Series, indexed by each window's
        LAST day. The first (lookback - 1) test rows cannot be predicted.
    """
    feats = test_df[artifacts.feature_columns]
    labels = test_df[label_column].astype(int)

    scaled = pd.DataFrame(
        artifacts.scaler.transform(feats),
        index=feats.index,
        columns=artifacts.feature_columns,
    )
    X, y = make_sequences(scaled, labels, lookback=artifacts.lookback)
    if len(X) == 0:
        raise ValueError(f"Test slice too small for lookback={artifacts.lookback}.")

    probs = artifacts.model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    pred_dates = test_df.index[artifacts.lookback - 1 :]
    return (
        pd.Series(preds, index=pred_dates, name="prediction"),
        pd.Series(y, index=pred_dates, name=label_column),
    )


def evaluate_gru(
    artifacts: GRUArtifacts,
    test_df: pd.DataFrame,
    label_column: str = "label_5d",
) -> EvalResult:
    """Evaluate a GRU on a test slice, reusing the shared metric set."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from src.features.target import BUY, SELL

    preds, y_true = predict_gru(artifacts, test_df, label_column=label_column)

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
    # Single train/test GRU run against cached MSFT data:
    #   python -m src.models.gru
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
    train_df = clean.loc[: split.split_date].iloc[:-1]
    test_df = clean.loc[split.split_date :]

    print(f"Training GRU on {len(train_df)} rows (lookback=60)...")
    artifacts = train_gru(train_df, lookback=60, verbose=1)

    print("\nEvaluating on test set...")
    result = evaluate_gru(artifacts, test_df)
    print(f"Accuracy:          {result.accuracy:.3f}")
    print(f"Majority baseline: {result.majority_baseline_accuracy:.3f}")
    print(f"Beats baseline?    {'YES' if result.beats_majority() else 'NO'}")
    print(f"F1 (BUY):          {result.f1:.3f}")
    print("\nConfusion matrix:")
    print(result.confusion)
    print("\nRead the confusion matrix, not just the accuracy: check whether the")
    print("GRU predicts both classes or collapses to one, as the others did.")