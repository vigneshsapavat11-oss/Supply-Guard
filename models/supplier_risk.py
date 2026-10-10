"""
SupplyGuard - Supplier Risk Prediction Module
Predicts supplier risk from operational features.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from utils.metrics import classification_metrics, supplier_risk_level
from config import MODELS_DIR, RANDOM_SEED

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class SupplierRiskPredictor:
    """
    Predicts supplier risk based on operational metrics.
    Creates synthetic risk labels from available supplier features.
    """
    
    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.results: Dict[str, Dict] = {}
        self.metrics: Dict = {}
        self.supplier_features: Optional[pd.DataFrame] = None
        
    def aggregate_supplier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transaction-level data to supplier-level features.
        """
        if "supplier_id" not in df.columns:
            raise ValueError("supplier_id column required for supplier risk analysis")
        
        agg_dict = {}
        
        if "lead_time" in df.columns:
            agg_dict["avg_lead_time"] = ("lead_time", "mean")
            agg_dict["lead_time_std"] = ("lead_time", "std")
        
        if "price" in df.columns:
            agg_dict["price_volatility"] = ("price", "std")
            agg_dict["avg_price"] = ("price", "mean")
        
        if "demand" in df.columns:
            agg_dict["avg_demand"] = ("demand", "mean")
            agg_dict["demand_std"] = ("demand", "std")
        
        if "inventory" in df.columns:
            agg_dict["avg_inventory"] = ("inventory", "mean")
        
        if not agg_dict:
            raise ValueError("Insufficient columns for supplier aggregation")
        
        supplier_df = df.groupby("supplier_id").agg(**agg_dict).reset_index()
        
        # Order frequency (number of transactions)
        order_counts = df.groupby("supplier_id").size().reset_index(name="order_frequency")
        supplier_df = supplier_df.merge(order_counts, on="supplier_id")
        
        # Derived features
        supplier_df["lead_time_variability"] = (
            supplier_df.get("lead_time_std", 0) / 
            (supplier_df.get("avg_lead_time", 1) + 1)
        )
        
        supplier_df["demand_volatility"] = (
            supplier_df.get("demand_std", 0) / 
            (supplier_df.get("avg_demand", 1) + 1)
        )
        
        # Fill NaN
        supplier_df = supplier_df.fillna(0)
        
        return supplier_df
    
    def create_risk_label(self, supplier_df: pd.DataFrame) -> pd.Series:
        """
        Create synthetic risk label based on operational metrics.
        This is a transparent rule-based label for training demonstration.
        
        Risk = HIGH if:
        - Lead time is above median AND
        - Lead time variability is above median
        
        This is a SIMULATED label for demonstration - not real supplier risk data.
        """
        lead_time = supplier_df.get("avg_lead_time", pd.Series([0] * len(supplier_df)))
        variability = supplier_df.get("lead_time_variability", pd.Series([0] * len(supplier_df)))
        demand_vol = supplier_df.get("demand_volatility", pd.Series([0] * len(supplier_df)))
        
        # Normalize each component to 0-1
        def normalize(s):
            if s.max() == s.min():
                return pd.Series([0.5] * len(s), index=s.index)
            return (s - s.min()) / (s.max() - s.min())
        
        risk_score = (
            normalize(lead_time) * 0.4 +
            normalize(variability) * 0.3 +
            normalize(demand_vol) * 0.3
        )
        
        # Binary label: 1 if risk score > 0.5
        return (risk_score > 0.5).astype(int)
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available classification models."""
        models = {
            "Logistic Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
            ]),
            "Random Forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=100, max_depth=8, random_state=RANDOM_SEED,
                    n_jobs=-1, class_weight="balanced"
                ))
            ]),
            "Gradient Boosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=RANDOM_SEED
                ))
            ]),
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(
                    n_estimators=100, max_depth=4, random_state=RANDOM_SEED,
                    verbosity=0
                ))
            ])
        
        return models
    
    def train(self, df: pd.DataFrame, progress_callback=None) -> Dict[str, Dict]:
        """Train supplier risk models."""
        supplier_df = self.aggregate_supplier_features(df)
        supplier_df["risk_label"] = self.create_risk_label(supplier_df)
        
        self.supplier_features = supplier_df
        
        # Features for training
        feature_cols = [c for c in supplier_df.columns 
                       if c not in ["supplier_id", "risk_label"]
                       and supplier_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        X = supplier_df[feature_cols].fillna(0)
        y = supplier_df["risk_label"]
        
        self.feature_names = feature_cols
        
        if y.nunique() < 2:
            return {"error": "Insufficient class variation for supplier risk training."}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
        )
        
        available_models = self.get_available_models()
        results = {}
        
        for i, (name, model) in enumerate(available_models.items()):
            if progress_callback:
                progress_callback(i / len(available_models), f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                metrics = classification_metrics(y_test, y_pred, y_prob)
                results[name] = {"model": model, "metrics": metrics}
            except Exception as e:
                results[name] = {"error": str(e), "metrics": {}}
        
        self.results = results
        
        valid = {k: v for k, v in results.items() if "metrics" in v and v["metrics"].get("F1")}
        if valid:
            self.model_name = max(valid.keys(), key=lambda k: valid[k]["metrics"]["F1"])
            self.model = self.results[self.model_name]["model"]
            self.metrics = self.results[self.model_name]["metrics"]
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return results
    
    def predict_supplier_risk(self) -> pd.DataFrame:
        """Predict risk for all suppliers."""
        if self.model is None or self.supplier_features is None:
            raise ValueError("Model not trained.")
        
        X = self.supplier_features[self.feature_names].fillna(0)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        result = self.supplier_features.copy()
        result["risk_probability"] = probabilities
        result["risk_category"] = result["risk_probability"].apply(supplier_risk_level)
        
        # Mitigation recommendations
        result["mitigation"] = result["risk_category"].map({
            "LOW": "Continue regular monitoring",
            "MEDIUM": "Review contract terms and delivery schedules",
            "HIGH": "Identify alternate suppliers, increase safety stock",
            "CRITICAL": "Immediate supplier audit, activate contingency plan"
        })
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if self.model is None:
            return pd.DataFrame()
        
        try:
            model_step = self.model.named_steps.get("model", self.model)
            if hasattr(model_step, "feature_importances_"):
                importance = model_step.feature_importances_
            elif hasattr(model_step, "coef_"):
                importance = np.abs(model_step.coef_[0])
            else:
                return pd.DataFrame()
            
            return pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": importance
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
    
    def save(self, path_prefix: str = None):
        """Save model."""
        if self.model is None:
            return
        prefix = path_prefix or str(MODELS_DIR / "supplier_model")
        joblib.dump(self.model, f"{prefix}.pkl")
        
        import json
        metadata = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "training_timestamp": datetime.now().isoformat()
        }
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)