from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Quieten TF's startup logging before import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


import tensorflow as tf  
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (  
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential

from src.models.random_forest import FEATURE_COLUMNS, EvalResult

from src.models.sequences import make_sequences

DEFAULT_LOOKBACK = 60
DEFAULT_SEED = 42

#Convolutional hyperparameters. kept small for same reason tthe LSTM is small: to avoid overfitting on noisy financial data.
DEFAULT_FILTERS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_POOL_SIZE = 2

def set_seeds(seed: int = DEFAULT_SEED) -> None:
    # set seeds fro as much reproducibility as TF allows.
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_cnn_lstm(
        lookback: int,
        n_features: int,
        filters: int = DEFAULT_FILTERS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        pool_size: int = DEFAULT_POOL_SIZE,
        seed: int = DEFAULT_SEED,
) -> Sequential : 
    """ 
    Build a compact CNN-LSTM hybrid classifier.
    Architecture:
        Conv1D      -> learns local patterns across neighbouring days
        MaxPooling1D-> halves the sequence length, keeping the strongest signals
        LSTM        -> models how the extracted features evolve over time
        Dropout     -> regularisation
        Dense       -> classification head, single sigmoid (BUY probability)
        
        LSTM layer in CNN_LSTM intentionality same size (32-units) as in lstm.py
        so the comparison isolates the effect of the convolutional stage rather than
        confusing it with a change in recurrent capacity.
    """         
    set_seeds(seed)

    # Guard: pooling must not consume the whole sequence.
    if lookback < kernel_size * pool_size:
        raise ValueError(
            f"lookback = {lookback} is too short for kernel_size = {kernel_size}"
            f"and pool_size = {pool_size}. Increase lookback or shrink the"
            f"convolution."
        )
    
    model = Sequential(
        [
            Input(shape = (lookback, n_features)),
            Conv1D(
                filters = filters,
                kernel_size = kernel_size,
                activation = "relu",
                padding = "causal",
            ),
            MaxPooling1D(pool_size = pool_size),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation = "relu"),
            Dense(1, activation = "sigmoid"),
        ]
    )

    model.compile (optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model


@dataclass
class CNNLSTMartifacts:
        #A trained CNN-LSTM model plus the scaler fit on its training data.

        model: Sequential
        scaler: StandardScaler
        lookback: int
        feature_columns: list[str]


def train_cnn_lstm(
    train_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "label_5d",
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> CNNLSTMArtifacts:
    """
    Train a CNN-LSTM on a training slice (leakage-safe scaling + windowing).
    Mirrors train_lstm exactly apart from the model builder, so the two are
    directly comparable. 
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
 
    model = build_cnn_lstm(lookback, len(feature_columns), seed=seed)
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
    return CNNLSTMartifacts(
        model=model,
        scaler=scaler,
        lookback=lookback,
        feature_columns=list(feature_columns),
    )

def predict_cnn_lstm(
    artifacts: CNNLSTMArtifacts,
    test_df: pd.DataFrame,
    label_column: str = "label_5d",
    threshold: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
     """
     Predict BUY/SELL on a test slice using a trained CNN-LSTM
     Test features are scaled with Training scaler. and windowed
     separately so no window straddles the train/test boundary.

     Its return:
        (predictions, true_labels) as aligned series, indexed by each window's
        Last day. The first (lookback - 1) test rows cannot be predicted
     """

     feats = test_df[artifacts.feature_columns]
     labels = test_df[label_column].astype(int)

     scaled = pd.DataFrame(
          artifacts.scaler.transform(feats),
          index = feats.index,
          columns = artifacts.feature_columns,
     )
     X, y = make_sequences(scaled, labels, lookback = artifacts.lookback)
     if len(X) == 0:
          raise ValueError(
               f"Test slice too small for lookback={artifacts.lookback}."
          )
     
     probs = artifacts.model.predict(X, verbose = 0).ravel()
     preds = (probs >= threshold).astype(int)

     pred_dates = test_df.index[artifacts.lookback - 1 :]
     return (
        pd.Series(preds, index=pred_dates, name="prediction"),
        pd.Series(y, index=pred_dates, name=label_column),
     )
        

def evaluate_cnn_lstm(
    artifacts: CNNLSTMartifacts,
    test_df: pd.DataFrame,
    label_column: str = "label_5d",
) -> EvalResult:
    #Evaluate a CNN-LSTM on a test slice, reusing the shared metric set.
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
 
    from src.features.target import BUY, SELL
 
    preds, y_true = predict_cnn_lstm(artifacts, test_df, label_column=label_column)
 
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
    # Single train/test CNN-LSTM run against cached MSFT data:
    #   python -m src.models.cnn_lstm
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
 
    print(f"Training CNN-LSTM on {len(train_df)} rows (lookback=60)...")
    artifacts = train_cnn_lstm(train_df, lookback=60, verbose=1)
 
    print("\nEvaluating on test set...")
    result = evaluate_cnn_lstm(artifacts, test_df)
    print(f"Accuracy:          {result.accuracy:.3f}")
    print(f"Majority baseline: {result.majority_baseline_accuracy:.3f}")
    print(f"Beats baseline?    {'YES' if result.beats_majority() else 'NO'}")
    print(f"F1 (BUY):          {result.f1:.3f}")
    print("\nConfusion matrix:")
    print(result.confusion)
    print("\nReminder: the CNN-LSTM may NOT beat the LSTM, the RF, or the")
    print("baseline. That is a legitimate, reportable finding — do not tune")
    print("it repeatedly against the test set until it looks better.")



    