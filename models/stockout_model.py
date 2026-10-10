"""
SupplyGuard - Stock-Out Prediction Module
Binary classification for stock-out risk.
"""

import pandas as pd
import numpy as np
import joblib
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from utils.metrics import classification_metrics, stockout_risk_level
from config import MODELS_DIR, RANDOM_SEED

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class StockoutPredictor:
    """
    Predicts stock-out probability for products.
    Creates the target from inventory/demand/lead_time relationships.
    """
    
    def __init__(self, target_col: str = "demand"):
        self.target_col = target_col
        self.model = None
        self.model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.results: Dict[str, Dict] = {}
        self.metrics: Dict = {}
        self.threshold = 0.5
        self.label_source = ""
        
    def create_stockout_label(self, df: pd.DataFrame,
                               horizon_days: int = 7) -> pd.Series:
        """
        Create binary stock-out label.
        Stock-out = 1 if inventory is insufficient to meet demand over the horizon.
        
        Label logic:
        - Compute expected demand over horizon (using rolling mean)
        - If inventory < expected demand * safety_factor => stock-out
        """
        if "inventory" not in df.columns:
            raise ValueError("Inventory column required for stock-out prediction")
        
        # Use rolling mean demand as expected daily demand
        if "rolling_mean_7" in df.columns:
            expected_daily = df["rolling_mean_7"].fillna(df[self.target_col])
        else:
            expected_daily = df[self.target_col]
        
        # Expected demand over horizon
        expected_horizon = expected_daily * horizon_days
        
        # Stock-out if inventory < expected demand (with small buffer)
        stockout = (df["inventory"] < expected_horizon * 0.8).astype(int)
        
        return stockout
    
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
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"
                ))
            ]),
            "Gradient Boosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    random_state=RANDOM_SEED
                ))
            ]),
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=RANDOM_SEED, verbosity=0,
                    scale_pos_weight=1
                ))
            ])
        
        return models
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for stock-out prediction."""
        exclude = ["date", self.target_col, "product_id", "warehouse", 
                   "supplier_id", "category", "stockout_label"]
        
        feature_cols = [c for c in df.columns if c not in exclude 
                       and not c.endswith("_encoded")
                       and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df.get("stockout_label")
        
        return X, y
    
    def train_all(self, df: pd.DataFrame, horizon_days: int = 7,
                  progress_callback=None) -> Dict[str, Dict]:
        """Train all classification models."""
        df = df.copy()
        if "stockout_label" not in df.columns:
            df["stockout_label"] = self.create_stockout_label(df, horizon_days)
            self.label_source = "inventory-demand rule"
        else:
            df["stockout_label"] = pd.to_numeric(df["stockout_label"], errors="coerce")
            labels = set(df["stockout_label"].dropna().unique())
            if not labels.issubset({0, 1}):
                return {"error": "stockout_label must contain binary 0/1 values."}
            self.label_source = "provided stockout_label"
            if len(labels) < 2:
                derived_labels = self.create_stockout_label(df, horizon_days)
                if derived_labels.nunique() < 2:
                    return {"error": "Provided stockout_label has one class and the inventory-demand rule also produces one class."}
                df["stockout_label"] = derived_labels
                self.label_source = "inventory-demand rule (provided label had one class)"
        
        # Check class balance
        class_counts = df["stockout_label"].value_counts()
        if len(class_counts) < 2:
            return {"error": "Only one class present in stock-out labels. Cannot train classifier."}
        
        X, y = self.prepare_features(df)
        
        # Remove rows with NaN target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        self.feature_names = X.columns.tolist()
        
        # Use stratified split (not time-based here since we're predicting current risk)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        available_models = self.get_available_models()
        results = {}
        
        for i, (name, model) in enumerate(available_models.items()):
            if progress_callback:
                progress_callback(i / len(available_models), f"Training {name}...")
            
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                metrics = classification_metrics(y_test, y_pred, y_prob)
                
                results[name] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": round(train_time, 3),
                    "y_test": y_test.values,
                    "y_prob": y_prob
                }
            except Exception as e:
                results[name] = {"error": str(e), "metrics": {}}
        
        self.results = results
        
        # Select best by F1 score (more appropriate for imbalanced data)
        valid = {k: v for k, v in results.items() 
                if "metrics" in v and v["metrics"].get("F1") is not None
                and not np.isnan(v["metrics"].get("F1", np.nan))}
        
        if valid:
            self.model_name = max(valid.keys(), key=lambda k: valid[k]["metrics"]["F1"])
            self.model = self.results[self.model_name]["model"]
            self.metrics = self.results[self.model_name]["metrics"]
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return results
    
    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict stock-out risk for each row."""
        if self.model is None:
            raise ValueError("No trained model available.")
        
        X, _ = self.prepare_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        result = df.copy()
        result["stockout_probability"] = probabilities
        result["risk_level"] = result["stockout_probability"].apply(stockout_risk_level)
        
        # Calculate days of inventory
        if "inventory" in result.columns and "rolling_mean_7" in result.columns:
            result["days_of_inventory"] = np.where(
                result["rolling_mean_7"] > 0,
                result["inventory"] / result["rolling_mean_7"],
                999
            )
        
        # Recommended action
        result["recommended_action"] = result["risk_level"].map({
            "LOW": "Monitor regularly",
            "MEDIUM": "Consider reordering soon",
            "HIGH": "Reorder immediately",
            "CRITICAL": "Emergency replenishment required"
        })
        
        return result
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get model comparison table."""
        rows = []
        for name, result in self.results.items():
            if "error" in result:
                continue
            m = result["metrics"]
            rows.append({
                "Model": name,
                "Accuracy": m.get("Accuracy"),
                "Precision": m.get("Precision"),
                "Recall": m.get("Recall"),
                "F1": m.get("F1"),
                "ROC_AUC": m.get("ROC_AUC"),
                "Training Time (s)": result["training_time"]
            })
        return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    
    def save(self, path_prefix: str = None):
        """Save the model."""
        if self.model is None:
            return
        prefix = path_prefix or str(MODELS_DIR / "stockout_model")
        joblib.dump(self.model, f"{prefix}.pkl")
        
        import json
        metadata = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "training_timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED
        }
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)