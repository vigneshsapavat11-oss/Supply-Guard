"""Tests for reproducible research experiment utilities."""

import pandas as pd

from utils.research import dataset_fingerprint, make_experiment_split, prepare_experiment_features


def test_chronological_split_keeps_dates_together_and_orders_test_period():
    dates = pd.Series(pd.date_range("2024-01-01", periods=10).repeat(2))

    train_indices, test_indices, strategy = make_experiment_split(
        len(dates), test_size=0.2, random_state=42, dates=dates
    )

    assert strategy == "chronological"
    assert dates.iloc[train_indices].max() < dates.iloc[test_indices].min()
    assert not set(train_indices).intersection(test_indices)


def test_random_split_is_reproducible_and_stratified_when_possible():
    labels = pd.Series([0, 1] * 20)

    first = make_experiment_split(40, 0.25, 7, stratify=labels)
    second = make_experiment_split(40, 0.25, 7, stratify=labels)

    assert first[2] == "stratified_random"
    assert first[0].tolist() == second[0].tolist()
    assert first[1].tolist() == second[1].tolist()
    assert labels.iloc[first[0]].value_counts().to_dict() == {0: 15, 1: 15}


def test_random_split_falls_back_when_stratification_cannot_fit_test_set():
    labels = pd.Series([0, 1] * 5)

    train_indices, test_indices, strategy = make_experiment_split(
        10, test_size=0.1, random_state=7, stratify=labels
    )

    assert strategy == "random"
    assert len(train_indices) == 9
    assert len(test_indices) == 1


def test_dataset_fingerprint_is_stable_and_changes_with_data():
    original = pd.DataFrame({"demand": [10, 20], "date": pd.date_range("2024-01-01", periods=2)})

    assert dataset_fingerprint(original) == dataset_fingerprint(original.copy())
    changed = original.assign(demand=[10, 21])
    assert dataset_fingerprint(original) != dataset_fingerprint(changed)


def test_price_experiment_uses_encoded_categories_and_calendar_features():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "product_id_encoded": [0, 1, 0],
        "market_encoded": [2, 2, 1],
        "price": [10.0, 11.0, 12.0],
    })

    features = prepare_experiment_features(df, "price")

    assert "product_id_encoded" in features
    assert "market_encoded" in features
    assert "date_year" in features
    assert "date_month" in features
    assert "price" not in features