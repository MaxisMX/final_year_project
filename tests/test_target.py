"""
Tests for src.features.target.

The headline test (`test_label_hand_verified`) uses a tiny price series where
every label can be worked out by hand, so the leakage-safe labelling is proven,
not assumed. The other tests guard the warm-down NaNs and the drop step that
people commonly forget.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.target import (
    BUY,
    SELL,
    attach_label,
    drop_unlabelled,
    forward_return,
    make_label,
)


def test_label_hand_verified() -> None:
    """
    Hand-worked example with horizon=2.

    Prices:   index 0..5 = [10, 11, 9, 12, 8, 13]
    horizon=2, so label(t) compares close(t+2) vs close(t):
      t=0: close(2)=9  vs 10 -> SELL (9 < 10)
      t=1: close(3)=12 vs 11 -> BUY  (12 > 11)
      t=2: close(4)=8  vs 9  -> SELL (8 < 9)
      t=3: close(5)=13 vs 12 -> BUY  (13 > 12)
      t=4: no close(6) -> NaN
      t=5: no close(7) -> NaN
    """
    close = pd.Series([10, 11, 9, 12, 8, 13], dtype=float)
    labels = make_label(close, horizon=2)

    assert labels.iloc[0] == SELL
    assert labels.iloc[1] == BUY
    assert labels.iloc[2] == SELL
    assert labels.iloc[3] == BUY
    assert pd.isna(labels.iloc[4])
    assert pd.isna(labels.iloc[5])


def test_last_horizon_rows_are_nan() -> None:
    """The final `horizon` rows must have no label there is no future for them."""
    close = pd.Series(range(1, 51), dtype=float)
    for h in (1, 5, 10):
        labels = make_label(close, horizon=h)
        assert labels.iloc[-h:].isna().all(), f"horizon={h}: last {h} rows should be NaN"
        assert labels.iloc[: -h].notna().all(), f"horizon={h}: earlier rows should be labelled"


def test_forward_return_values() -> None:
    close = pd.Series([100.0, 110.0, 121.0], dtype=float)
    fr = forward_return(close, horizon=1)
    # t=0: 110/100 - 1 = 0.10 ; t=1: 121/110 - 1 = 0.10 ; t=2: NaN
    assert fr.iloc[0] == pytest.approx(0.10)
    assert fr.iloc[1] == pytest.approx(0.10)
    assert pd.isna(fr.iloc[2])


def test_flat_price_is_sell() -> None:
    """Zero forward return counts as SELL (not strictly greater than today)."""
    close = pd.Series([50.0, 50.0, 50.0, 50.0], dtype=float)
    labels = make_label(close, horizon=1)
    assert labels.iloc[0] == SELL
    assert labels.iloc[1] == SELL


def test_attach_label_does_not_mutate() -> None:
    df = pd.DataFrame({"Close": range(1, 30)}, dtype=float)
    cols_before = df.columns.tolist()
    out = attach_label(df, horizon=5)
    assert df.columns.tolist() == cols_before
    assert "label_5d" in out.columns


def test_drop_unlabelled_removes_exactly_horizon_rows() -> None:
    df = pd.DataFrame({"Close": range(1, 31)}, dtype=float)  # 30 rows
    labelled = attach_label(df, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5)
    assert len(clean) == 25  # 30 - 5
    assert clean["label_5d"].notna().all()


def test_drop_unlabelled_requires_label_column() -> None:
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="No label column"):
        drop_unlabelled(df, horizon=5)


def test_labels_are_only_zero_or_one_after_drop() -> None:
    df = pd.DataFrame({"Close": range(1, 100)}, dtype=float)
    labelled = attach_label(df, horizon=5)
    clean = drop_unlabelled(labelled, horizon=5)
    assert set(clean["label_5d"].unique()).issubset({BUY, SELL})