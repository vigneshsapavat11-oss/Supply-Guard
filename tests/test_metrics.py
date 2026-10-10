"""Tests for metrics module."""

import pytest
import numpy as np
from utils.metrics import (
    regression_metrics, classification_metrics,
    stockout_risk_level, supplier_risk_level,
    inventory_status, calculate_safety_stock
)


def test_regression_metrics():
    """Test regression metrics calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])
    
    metrics = regression_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] > 0


def test_scale_independent_forecasting_metrics():
    metrics = regression_metrics(
        np.array([2, 4]), np.array([1, 4]), y_train=np.array([1, 2, 3])
    )

    assert metrics["WAPE"] == 16.67
    assert metrics["sMAPE"] == 33.33
    assert metrics["MASE"] == 0.5


def test_classification_metrics():
    """Test classification metrics calculation."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.6, 0.9])
    
    metrics = classification_metrics(y_true, y_pred, y_prob)
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1" in metrics
    assert "ROC_AUC" in metrics


def test_stockout_risk_level():
    """Test stock-out risk level classification."""
    assert stockout_risk_level(0.1) == "LOW"
    assert stockout_risk_level(0.3) == "MEDIUM"
    assert stockout_risk_level(0.6) == "HIGH"
    assert stockout_risk_level(0.9) == "CRITICAL"


def test_supplier_risk_level():
    """Test supplier risk level classification."""
    assert supplier_risk_level(0.1) == "LOW"
    assert supplier_risk_level(0.4) == "MEDIUM"
    assert supplier_risk_level(0.7) == "HIGH"
    assert supplier_risk_level(0.95) == "CRITICAL"


def test_inventory_status():
    """Test inventory status classification."""
    assert inventory_status(1, 5) == "CRITICAL STOCK"
    assert inventory_status(3, 5) == "LOW STOCK"
    assert inventory_status(8, 5) == "HEALTHY"
    assert inventory_status(20, 5) == "OVERSTOCK"


def test_safety_stock():
    """Test safety stock calculation."""
    ss = calculate_safety_stock(demand_std=10, lead_time=5, service_level=0.95)
    assert ss > 0
    assert isinstance(ss, float)