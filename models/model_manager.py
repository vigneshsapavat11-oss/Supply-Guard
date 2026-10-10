"""
SupplyGuard - Model Manager
Central model registry and orchestration.
"""

import pandas as pd
from typing import Dict, Optional, Any
import streamlit as st
from models.demand_forecasting import DemandForecaster
from models.stockout_model import StockoutPredictor
from models.supplier_risk import SupplierRiskPredictor
from models.anomaly_detection import AnomalyDetector
from models.warehouse_optimization import WarehouseOptimizer


class ModelManager:
    """
    Central manager for all SupplyGuard models.
    Handles training, caching, and retrieval.
    """
    
    def __init__(self):
        self.demand_forecaster: Optional[DemandForecaster] = None
        self.stockout_predictor: Optional[StockoutPredictor] = None
        self.supplier_predictor: Optional[SupplierRiskPredictor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.warehouse_optimizer: Optional[WarehouseOptimizer] = None
        self.training_status: Dict[str, str] = {}
    
    def train_all(self, df: pd.DataFrame, progress_callback=None) -> Dict[str, Any]:
        """Train all models on the provided dataset."""
        self.demand_forecaster = None
        self.stockout_predictor = None
        self.supplier_predictor = None
        self.anomaly_detector = None
        self.warehouse_optimizer = None
        self.training_status = {}

        results = {}
        total_steps = 5
        current_step = 0
        
        # 1. Demand Forecasting
        target_candidates = ("demand", "price", "food_price", "inflation", "sales", "quantity", "units_sold")
        target_col = next(
            (column for column in target_candidates if column in df.columns and pd.api.types.is_numeric_dtype(df[column])),
            None,
        )
        date_col = next(
            (column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])),
            "date" if "date" in df.columns else None,
        )
        if target_col is None or date_col is None:
            requirements = []
            if target_col is None:
                requirements.append("a numeric demand, sales, quantity, or price column")
            if date_col is None:
                requirements.append("a date column")
            self.training_status["demand"] = f"unavailable: requires {' and '.join(requirements)}"
            results["demand"] = {"error": self.training_status["demand"]}
        else:
            self.training_status["demand"] = "training"
            try:
                self.demand_forecaster = DemandForecaster(target_col=target_col, date_col=date_col)
                results["demand"] = self.demand_forecaster.train_all(
                    df, progress_callback=lambda p, msg: progress_callback(
                        (current_step + p) / total_steps, f"Forecast: {msg}"
                    ) if progress_callback else None
                )
                if self.demand_forecaster.best_model is None:
                    self.training_status["demand"] = "error: no model completed validation"
                else:
                    self.training_status["demand"] = "complete"
            except Exception as e:
                self.training_status["demand"] = f"error: {e}"
                results["demand"] = {"error": str(e)}

        current_step += 1
        
        # 2. Stock-Out Prediction
        if not {"inventory", "demand"}.issubset(df.columns):
            self.training_status["stockout"] = "unavailable: requires demand and inventory"
            results["stockout"] = {"error": self.training_status["stockout"]}
        else:
            self.training_status["stockout"] = "training"
            try:
                self.stockout_predictor = StockoutPredictor()
                results["stockout"] = self.stockout_predictor.train_all(
                    df, progress_callback=lambda p, msg: progress_callback(
                        (current_step + p) / total_steps, f"Stock-Out: {msg}"
                    ) if progress_callback else None
                )
                if "error" in results["stockout"]:
                    self.training_status["stockout"] = f"unavailable: {results['stockout']['error']}"
                elif self.stockout_predictor.label_source != "provided stockout_label":
                    self.training_status["stockout"] = f"complete (rule-derived labels: {self.stockout_predictor.label_source})"
                else:
                    self.training_status["stockout"] = "complete"
            except Exception as e:
                self.training_status["stockout"] = f"error: {e}"
                results["stockout"] = {"error": str(e)}
        current_step += 1
        
        # 3. Supplier Risk
        if "supplier_id" not in df.columns:
            self.training_status["supplier"] = "unavailable: requires supplier_id"
            results["supplier"] = {"error": self.training_status["supplier"]}
        else:
            self.training_status["supplier"] = "training"
            try:
                self.supplier_predictor = SupplierRiskPredictor()
                results["supplier"] = self.supplier_predictor.train(
                    df, progress_callback=lambda p, msg: progress_callback(
                        (current_step + p) / total_steps, f"Supplier: {msg}"
                    ) if progress_callback else None
                )
                if "error" in results["supplier"]:
                    self.training_status["supplier"] = f"unavailable: {results['supplier']['error']}"
                else:
                    self.training_status["supplier"] = "complete (rule-derived supplier-risk labels)"
            except Exception as e:
                self.training_status["supplier"] = f"error: {e}"
                results["supplier"] = {"error": str(e)}
        current_step += 1
        
        # 4. Anomaly Detection
        if target_col is None:
            self.training_status["anomaly"] = "unavailable: requires dated demand, sales, or price data"
            results["anomaly"] = {"error": self.training_status["anomaly"]}
        else:
            self.training_status["anomaly"] = "training"
            try:
                self.anomaly_detector = AnomalyDetector()
                results["anomaly"] = self.anomaly_detector.run_all(df)
                self.training_status["anomaly"] = "complete"
            except Exception as e:
                self.training_status["anomaly"] = f"error: {e}"
                results["anomaly"] = {"error": str(e)}
        current_step += 1
        
        # 5. Warehouse Optimizer (no training needed)
        if {"product_id", "warehouse", "demand", "inventory"}.issubset(df.columns):
            self.warehouse_optimizer = WarehouseOptimizer()
            self.training_status["warehouse"] = "complete"
        else:
            self.warehouse_optimizer = None
            self.training_status["warehouse"] = (
                "unavailable: requires product_id, warehouse, demand, and inventory"
            )
        
        if progress_callback:
            progress_callback(1.0, "Training and availability checks finished")
        
        return results
    
    def get_status(self) -> Dict[str, str]:
        """Get training status of all models."""
        return self.training_status.copy()
    
    def is_trained(self) -> bool:
        """Check if all essential models are trained."""
        return bool(self.training_status) and all(
            status == "complete" or status.startswith("complete (")
            for status in self.training_status.values()
        )


@st.cache_resource(show_spinner=False)
def get_model_manager():
    """Get or create the global model manager."""
    return ModelManager()