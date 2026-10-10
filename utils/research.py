"""Utilities for reproducible research experiments."""

import hashlib
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_experiment_split(
    row_count: int,
    test_size: float,
    random_state: int,
    dates: Optional[pd.Series] = None,
    stratify: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return positional train/test indices, preferring a chronological split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if row_count < 2:
        raise ValueError("At least two rows are required for an experiment.")

    if dates is not None and len(dates) == row_count:
        parsed_dates = pd.to_datetime(dates, errors="coerce")
        valid_dates = parsed_dates.notna().to_numpy()
        unique_dates = np.sort(parsed_dates[valid_dates].unique())
        if valid_dates.all() and len(unique_dates) > 1:
            split_at = int(len(unique_dates) * (1 - test_size))
            split_at = min(max(split_at, 1), len(unique_dates) - 1)
            cutoff = unique_dates[split_at]
            train_indices = np.flatnonzero((parsed_dates < cutoff).to_numpy())
            test_indices = np.flatnonzero((parsed_dates >= cutoff).to_numpy())
            return train_indices, test_indices, "chronological"

    indices = np.arange(row_count)
    stratify_values = stratify
    if stratify_values is not None:
        counts = pd.Series(stratify_values).value_counts()
        test_count = math.ceil(row_count * test_size)
        train_count = row_count - test_count
        if (
            counts.min() < 2
            or test_count < len(counts)
            or train_count < len(counts)
        ):
            stratify_values = None

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    return train_indices, test_indices, "stratified_random" if stratify_values is not None else "random"


def prepare_experiment_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Select numeric predictors and derive calendar features from datetime columns."""
    excluded = {target_col, "date", "product_id", "warehouse", "supplier_id", "category"}
    numeric_columns = [
        column for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    features = df[numeric_columns].copy()

    for column in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        dates = pd.to_datetime(df[column], errors="coerce")
        features[f"{column}_year"] = dates.dt.year
        features[f"{column}_month"] = dates.dt.month
        features[f"{column}_quarter"] = dates.dt.quarter

    return features


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint for the experiment input table."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()