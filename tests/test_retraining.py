"""
Tests for rolling-window retraining data preparation.
"""

import pandas as pd

from src.retrain import prepare_retraining_data


def _make_batch(start_age: int, n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "AGEP": float(start_age + i),
            "WKHP": float(40 + i),
            "COW": 1.0,
            "SCHL": 16.0,
            "MAR": 1.0,
            "OCCP": 5000.0,
            "POBP": 12.0,
            "RELP": 1.0,
            "SEX": 1.0 if i % 2 == 0 else 2.0,
            "RAC1P": 1.0,
            "income": ">50K" if i % 2 == 0 else "<=50K",
        })
    return pd.DataFrame(rows)


def test_prepare_retraining_data_uses_recent_window_only():
    original_train = pd.concat([_make_batch(20, 10), _make_batch(40, 10)], ignore_index=True)
    recent_batches = [_make_batch(100), _make_batch(200), _make_batch(300), _make_batch(400)]
    thresholds = {
        "retraining_data": {
            "window_batches": 2,
            "anchor_fraction": 0.0,
        }
    }
    settings = {
        "data": {"target_column": "income"},
        "project": {"random_seed": 42},
    }

    retraining_df = prepare_retraining_data(recent_batches, original_train, thresholds, settings)

    assert retraining_df["AGEP"].min() >= 300
    assert retraining_df["AGEP"].max() < 500


def test_prepare_retraining_data_includes_anchor_sample():
    original_train = pd.concat([_make_batch(20, 20), _make_batch(60, 20)], ignore_index=True)
    recent_batches = [_make_batch(300, 4)]
    thresholds = {
        "retraining_data": {
            "window_batches": 1,
            "anchor_fraction": 0.1,
        }
    }
    settings = {
        "data": {"target_column": "income"},
        "project": {"random_seed": 42},
    }

    retraining_df = prepare_retraining_data(recent_batches, original_train, thresholds, settings)

    assert (retraining_df["AGEP"] < 100).any()
    assert (retraining_df["AGEP"] >= 300).any()
    assert set(retraining_df["income"].unique()) == {">50K", "<=50K"}


def test_prepare_retraining_data_anchor_is_stratified():
    original_train = pd.DataFrame({
        "AGEP": [20.0] * 8 + [60.0] * 8,
        "WKHP": [40.0] * 16,
        "COW": [1.0] * 16,
        "SCHL": [16.0] * 16,
        "MAR": [1.0] * 16,
        "OCCP": [5000.0] * 16,
        "POBP": [12.0] * 16,
        "RELP": [1.0] * 16,
        "SEX": [1.0] * 16,
        "RAC1P": [1.0] * 16,
        "income": [">50K"] * 8 + ["<=50K"] * 8,
    })
    recent_batches = [_make_batch(300, 4)]
    thresholds = {
        "retraining_data": {
            "window_batches": 1,
            "anchor_fraction": 0.125,
        }
    }
    settings = {
        "data": {"target_column": "income"},
        "project": {"random_seed": 42},
    }

    retraining_df = prepare_retraining_data(recent_batches, original_train, thresholds, settings)
    anchor_rows = retraining_df[retraining_df["AGEP"] < 100]

    assert set(anchor_rows["income"].unique()) == {">50K", "<=50K"}
