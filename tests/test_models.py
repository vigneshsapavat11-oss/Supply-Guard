"""Tests for ML models."""

import pytest
import pandas as pd
import numpy as np
from models.demand_forecasting import DemandForecaster
from models.stockout_model import StockoutPredictor
from models.anomaly_detection import AnomalyDetector
from models.model_manager import ModelManager


def create_training_data(n=200):
    """Create synthetic training data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    demand = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 30) + np.random.normal(0, 10, n)
    
    return pd.DataFrame({
        "date": dates,
        "demand": np.maximum(demand, 0),
        "inventory": np.random.randint(50, 500, n),
        "price": np.random.uniform(10, 50, n),
        "promotion": np.random.randint(0, 2, n),
        "lead_time": np.random.uniform(1, 10, n),
        "lag_1": np.roll(demand, 1),
        "lag_7": np.roll(demand, 7),
        "rolling_mean_7": pd.Series(demand).rolling(7, min_periods=1).mean(),
        "rolling_std_7": pd.Series(demand).rolling(7, min_periods=1).std().fillna(0),
        "product_id": ["A"] * n,
        "warehouse": ["WH1"] * n,
    })


def test_demand_forecaster_init():
    """Test demand forecaster initialization."""
    forecaster = DemandForecaster()
    assert forecaster.target_col == "demand"
    assert forecaster.best_model is None


def test_chronological_split():
    """Test chronological data splitting."""
    df = create_training_data(100)
    forecaster = DemandForecaster()
    train, val, test = forecaster.chronological_split(df)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_chronological_split_keeps_duplicate_dates_together():
    df = create_training_data(100)
    df = pd.concat([df, df.iloc[[0]].assign(demand=999)], ignore_index=True)
    forecaster = DemandForecaster()

    train, validation, test = forecaster.chronological_split(df)

    train_dates = set(train["date"])
    validation_dates = set(validation["date"])
    test_dates = set(test["date"])
    assert not train_dates.intersection(validation_dates)
    assert not train_dates.intersection(test_dates)
    assert not validation_dates.intersection(test_dates)


def test_demand_feature_engineering_uses_only_past_target_values():
    df = create_training_data(40)
    forecaster = DemandForecaster()

    engineered = forecaster.engineer_features(df)
    features, _ = forecaster.prepare_features(engineered)

    assert pd.isna(engineered.loc[0, "lag_1"])
    assert engineered.loc[1, "lag_1"] == engineered.loc[0, "demand"]
    assert pd.isna(engineered.loc[0, "rolling_mean_7"])
    assert "inventory" not in features.columns


def test_demand_training():
    """Test demand model training."""
    df = create_training_data(200)
    forecaster = DemandForecaster()
    results = forecaster.train_all(df)
    assert len(results) > 0
    assert forecaster.best_model is not None
    assert "LightGBM" in results
    assert not results["LightGBM"].get("error")
    assert results["LightGBM"]["cv_folds"] == 3
    assert "num_leaves" in results["LightGBM"]["params"]
    assert all("validation_metrics" in result for result in results.values())
    assert forecaster.best_model_name == min(
        results, key=lambda name: results[name]["validation_metrics"]["RMSE"]
    )
    forecast = forecaster.forecast_future(df, horizon=2)
    assert len(forecast) == 2
    assert "predicted_demand" in forecast.columns


def test_price_forecaster_predicts_monthly_periods():
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    df = pd.DataFrame({
        "date": dates,
        "price": 10 + np.arange(len(dates)) * 0.2,
        "commodity_encoded": np.zeros(len(dates), dtype=int),
        "market_encoded": np.ones(len(dates), dtype=int),
    })
    forecaster = DemandForecaster(target_col="price")

    results = forecaster.train_all(df)
    forecast = forecaster.forecast_future(df, horizon=2)

    assert forecaster.best_model is not None
    assert "predicted_price" in forecast.columns
    assert forecast["forecast_date"].tolist() == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]
    assert not results[forecaster.best_model_name].get("error")


def test_demand_comparison_table_when_all_models_fail():
    """Return an empty comparison table with stable columns on failures."""
    forecaster = DemandForecaster()
    forecaster.results = {
        "Linear Regression": {"error": "training failed", "test_metrics": {}, "training_time": 0}
    }

    comparison = forecaster.get_comparison_table()

    assert comparison.empty
    assert comparison.columns.tolist() == [
        "Model", "Validation RMSE", "Test RMSE", "MAE", "R2", "MAPE",
        "WAPE", "sMAPE", "MASE", "Training Time (s)"
    ]


def test_stockout_label_creation():
    """Test stock-out label creation."""
    df = create_training_data(100)
    predictor = StockoutPredictor()
    labels = predictor.create_stockout_label(df, horizon_days=7)
    assert len(labels) == len(df)
    assert set(labels.unique()).issubset({0, 1})


def test_stockout_training():
    """Test stock-out model training."""
    df = create_training_data(300)
    predictor = StockoutPredictor()
    results = predictor.train_all(df)
    assert isinstance(results, dict)


def test_stockout_training_uses_provided_binary_labels():
    df = create_training_data(300)
    df["stockout_label"] = np.tile([0, 1], 150)
    predictor = StockoutPredictor()

    results = predictor.train_all(df)

    assert results
    assert predictor.metrics
    assert predictor.label_source == "provided stockout_label"


def test_stockout_uses_rule_labels_when_provided_flag_has_one_class():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=300),
        "demand": np.full(300, 30),
        "inventory": np.tile([100, 2000], 150),
        "price": np.full(300, 10.0),
        "stockout_label": np.zeros(300, dtype=int),
    })
    predictor = StockoutPredictor()

    results = predictor.train_all(df)

    assert results
    assert predictor.metrics
    assert predictor.label_source == "inventory-demand rule (provided label had one class)"


def test_model_manager_marks_missing_real_dataset_modules_unavailable(monkeypatch):
    class FakeForecaster:
        def __init__(self, target_col, date_col):
            self.best_model = object()

        def train_all(self, df, progress_callback=None):
            return {"ok": True}

    class FakeAnomalyDetector:
        def run_all(self, df):
            return {"ok": True}

    monkeypatch.setattr("models.model_manager.DemandForecaster", FakeForecaster)
    monkeypatch.setattr("models.model_manager.AnomalyDetector", FakeAnomalyDetector)
    manager = ModelManager()
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "demand": [1, 2, 3]})

    manager.train_all(df)

    assert manager.training_status["demand"] == "complete"
    assert manager.training_status["anomaly"] == "complete"
    assert manager.training_status["stockout"].startswith("unavailable:")
    assert manager.training_status["supplier"].startswith("unavailable:")
    assert manager.training_status["warehouse"].startswith("unavailable:")


def test_model_manager_rejects_customer_profile_data_for_supply_chain_forecasts():
    manager = ModelManager()
    manager.demand_forecaster = object()
    manager.stockout_predictor = object()
    manager.supplier_predictor = object()
    manager.anomaly_detector = object()
    manager.warehouse_optimizer = object()
    manager.training_status = {"previous_dataset": "complete"}
    customer_profile = pd.DataFrame({
        "Age": [20, 25, 30],
        "Output": ["Yes", "No", "Yes"],
        "Feedback": ["Positive", "Negative", "Positive"],
    })

    manager.train_all(customer_profile)

    assert manager.training_status["demand"].startswith("unavailable:")
    assert manager.training_status["anomaly"].startswith("unavailable:")
    assert manager.training_status["stockout"].startswith("unavailable:")
    assert manager.training_status["supplier"].startswith("unavailable:")
    assert manager.training_status["warehouse"].startswith("unavailable:")
    assert len(manager.training_status) == 5
    assert manager.demand_forecaster is None
    assert manager.stockout_predictor is None
    assert manager.supplier_predictor is None
    assert manager.anomaly_detector is None
    assert manager.warehouse_optimizer is None


def test_model_manager_accepts_transparently_labeled_completed_modules():
    manager = ModelManager()
    manager.training_status = {
        "demand": "complete",
        "stockout": "complete (rule-derived labels)",
        "supplier": "complete (rule-derived supplier-risk labels)",
    }

    assert manager.is_trained()


def test_anomaly_detection():
    """Test anomaly detection."""
    df = create_training_data(100)
    detector = AnomalyDetector()
    results = detector.run_all(df)
    assert "isolation_forest" in results
    assert "zscore" in results
    assert "iqr" in results


def test_dataset_provided_demand_forecast_is_reported_as_baseline():
    forecaster = DemandForecaster()
    train = pd.DataFrame({"demand": [10, 11, 12]})
    test = pd.DataFrame({"demand": [20, 22], "demand_forecast": [19, 23]})

    forecaster.compute_baseline(train, test)

    assert "Dataset-provided demand forecast" in forecaster.baseline_metrics
    assert forecaster.baseline_metrics["Dataset-provided demand forecast"]["RMSE"] < 2