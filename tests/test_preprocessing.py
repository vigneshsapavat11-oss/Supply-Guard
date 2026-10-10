"""Tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
from utils.data_preprocessing import DataPreprocessor


def create_test_data():
    """Create a small test dataset."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "product_id": ["A"] * 100,
        "warehouse": ["WH1"] * 100,
        "demand": np.random.randint(10, 100, 100),
        "inventory": np.random.randint(0, 500, 100),
        "price": np.random.uniform(10, 50, 100),
        "lead_time": np.random.uniform(1, 10, 100),
    })


def test_preprocessor_init():
    """Test preprocessor initialization."""
    df = create_test_data()
    preprocessor = DataPreprocessor(df)
    assert preprocessor.raw_df is not None
    assert len(preprocessor.raw_df) == 100


def test_detect_schema():
    """Test schema detection."""
    df = create_test_data()
    preprocessor = DataPreprocessor(df)
    schema = preprocessor.detect_schema()
    assert "numerical" in schema
    assert "demand" in schema["numerical"]


def test_parse_dates():
    """Test date parsing."""
    df = create_test_data()
    preprocessor = DataPreprocessor(df)
    result = preprocessor.parse_dates()
    assert result is True
    assert preprocessor.date_column == "date"


def test_identify_target():
    """Test target identification."""
    df = create_test_data()
    preprocessor = DataPreprocessor(df)
    target = preprocessor.identify_target()
    assert target == "demand"


def test_remove_duplicates():
    """Test duplicate removal."""
    original = create_test_data()
    df = pd.concat([original, original.iloc[:10]])
    preprocessor = DataPreprocessor(df)
    result = preprocessor.remove_duplicates()
    assert len(result) == 100
    assert preprocessor.report["duplicates_removed"] == 10


def test_handle_missing_values():
    """Test missing value handling."""
    df = create_test_data()
    df.loc[0:5, "demand"] = np.nan
    preprocessor = DataPreprocessor(df)
    result = preprocessor.handle_missing_values()
    assert result["demand"].isnull().sum() == 0


def test_full_pipeline():
    """Test complete preprocessing pipeline."""
    df = create_test_data()
    preprocessor = DataPreprocessor(df)
    result, report = preprocessor.run_pipeline()
    assert report["rows_after"] > 0
    assert report["target_variable"] == "demand"


def test_supply_chain_header_aliases_are_normalized():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3),
        "SKU_ID": ["SKU_1"] * 3,
        "Warehouse_ID": ["WH_1"] * 3,
        "Supplier_ID": ["SUP_1"] * 3,
        "Units_Sold": [10, 12, 9],
        "Inventory_Level": [100, 88, 79],
        "Supplier_Lead_Time_Days": [4, 4, 4],
        "Unit_Price": [2.5, 2.5, 2.6],
        "Stockout_Flag": [0, 0, 1],
        "Demand_Forecast": [11, 10, 9],
    })
    preprocessor = DataPreprocessor(df)

    processed, report = preprocessor.run_pipeline()

    assert report["target_variable"] == "demand"
    assert {"date", "product_id", "warehouse", "supplier_id", "demand", "inventory", "lead_time", "price", "stockout_label"}.issubset(processed.columns)