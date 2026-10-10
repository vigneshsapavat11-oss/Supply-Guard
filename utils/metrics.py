"""
SupplyGuard - Metrics Module
Proper regression and classification metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    Note: We do NOT call 1 - MAE/mean "accuracy" - that's not a valid metric.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan,
            "MAPE": np.nan, "WAPE": np.nan, "sMAPE": np.nan, "MASE": np.nan,
        }
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (only where y_true != 0)
    mask_mape = y_true != 0
    if mask_mape.sum() > 0:
        mape = np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100
    else:
        mape = np.nan

    absolute_errors = np.abs(y_true - y_pred)
    wape_denominator = np.abs(y_true).sum()
    wape = absolute_errors.sum() / wape_denominator * 100 if wape_denominator else np.nan

    smape_denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.divide(
        2 * absolute_errors,
        smape_denominator,
        out=np.zeros_like(absolute_errors, dtype=float),
        where=smape_denominator != 0,
    )
    smape = np.mean(smape_terms) * 100

    mase = np.nan
    if y_train is not None:
        y_train = np.asarray(y_train, dtype=float)
        y_train = y_train[np.isfinite(y_train)]
        if len(y_train) > 1:
            naive_scale = np.mean(np.abs(np.diff(y_train)))
            if naive_scale > 0:
                mase = mae / naive_scale
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2) if not np.isnan(mape) else np.nan,
        "WAPE": round(wape, 2) if not np.isnan(wape) else np.nan,
        "sMAPE": round(smape, 2),
        "MASE": round(mase, 4) if not np.isnan(mase) else np.nan,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    For imbalanced data, we report precision, recall, F1, and ROC-AUC.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    
    if y_prob is not None:
        try:
            metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    
    metrics["Confusion_Matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def get_classification_report(y_true, y_pred) -> str:
    """Get full classification report as string."""
    return classification_report(y_true, y_pred, zero_division=0)


def stockout_risk_level(probability: float) -> str:
    """Classify stock-out risk probability into risk level."""
    if probability < 0.25:
        return "LOW"
    elif probability < 0.50:
        return "MEDIUM"
    elif probability < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


def supplier_risk_level(probability: float) -> str:
    """Classify supplier risk probability into risk level."""
    if probability < 0.25:
        return "LOW"
    elif probability < 0.50:
        return "MEDIUM"
    elif probability < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


def inventory_status(days_of_inventory: float, lead_time: float) -> str:
    """Classify inventory status based on days of inventory vs lead time."""
    if days_of_inventory < lead_time * 0.5:
        return "CRITICAL STOCK"
    elif days_of_inventory < lead_time:
        return "LOW STOCK"
    elif days_of_inventory < lead_time * 3:
        return "HEALTHY"
    else:
        return "OVERSTOCK"


def calculate_safety_stock(demand_std: float, lead_time: float,
                           service_level: float = 0.95) -> float:
    """
    Calculate safety stock using the standard formula.
    safety_stock = z * sigma_demand * sqrt(lead_time)
    """
    from scipy.stats import norm
    z = norm.ppf(service_level)
    return z * demand_std * np.sqrt(lead_time)


def calculate_reorder_point(avg_demand: float, lead_time: float,
                            safety_stock: float) -> float:
    """Calculate reorder point."""
    return avg_demand * lead_time + safety_stock