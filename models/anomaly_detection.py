"""
SupplyGuard - Demand Anomaly Detection Module
Implements Isolation Forest and statistical baselines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import RANDOM_SEED


class AnomalyDetector:
    """
    Detect demand anomalies using multiple methods.
    """
    
    def __init__(self, target_col: str = "demand"):
        self.target_col = target_col
        self.results: Dict[str, pd.DataFrame] = {}
        self.comparison: Dict[str, Dict] = {}
    
    def detect_isolation_forest(self, df: pd.DataFrame,
                                 contamination: float = 0.05) -> pd.DataFrame:
        """
        Isolation Forest anomaly detection.
        """
        result = df.copy()
        
        feature_cols = [self.target_col]
        if "rolling_mean_7" in df.columns:
            feature_cols.append("rolling_mean_7")
        if "rolling_std_7" in df.columns:
            feature_cols.append("rolling_std_7")
        if "lag_1" in df.columns:
            feature_cols.append("lag_1")
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_estimators=100
        )
        
        labels = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.decision_function(X_scaled)
        
        result["anomaly_label_if"] = labels  # -1 = anomaly, 1 = normal
        result["anomaly_score_if"] = -scores  # Higher = more anomalous
        
        # Classify anomaly type
        result["anomaly_type_if"] = "Normal"
        median_demand = df[self.target_col].median()
        
        anomaly_mask = labels == -1
        result.loc[anomaly_mask & (result[self.target_col] > median_demand * 1.5), "anomaly_type_if"] = "Spike"
        result.loc[anomaly_mask & (result[self.target_col] < median_demand * 0.5), "anomaly_type_if"] = "Drop"
        result.loc[anomaly_mask & (result["anomaly_type_if"] == "Normal"), "anomaly_type_if"] = "Unusual"
        
        return result
    
    def detect_zscore(self, df: pd.DataFrame, threshold: float = 3.0,
                      window: int = 30) -> pd.DataFrame:
        """
        Rolling Z-score based anomaly detection.
        """
        result = df.copy()
        
        grouped = result.groupby([c for c in ["product_id", "warehouse"] if c in result.columns],
                                  group_keys=False)
        
        if len([c for c in ["product_id", "warehouse"] if c in result.columns]) > 0:
            rolling_mean = grouped[self.target_col].transform(
                lambda x: x.rolling(window, min_periods=5).mean()
            )
            rolling_std = grouped[self.target_col].transform(
                lambda x: x.rolling(window, min_periods=5).std()
            )
        else:
            rolling_mean = result[self.target_col].rolling(window, min_periods=5).mean()
            rolling_std = result[self.target_col].rolling(window, min_periods=5).std()
        
        z_scores = np.where(
            rolling_std > 0,
            (result[self.target_col] - rolling_mean) / rolling_std,
            0
        )
        
        result["zscore"] = z_scores
        result["anomaly_label_zscore"] = np.where(np.abs(z_scores) > threshold, -1, 1)
        result["anomaly_score_zscore"] = np.abs(z_scores)
        
        return result
    
    def detect_iqr(self, df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
        """
        IQR-based outlier detection.
        """
        result = df.copy()
        
        Q1 = result[self.target_col].quantile(0.25)
        Q3 = result[self.target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        
        result["anomaly_label_iqr"] = np.where(
            (result[self.target_col] < lower) | (result[self.target_col] > upper),
            -1, 1
        )
        
        return result
    
    def run_all(self, df: pd.DataFrame, contamination: float = 0.05) -> Dict[str, pd.DataFrame]:
        """Run all anomaly detection methods."""
        self.results["isolation_forest"] = self.detect_isolation_forest(df, contamination)
        self.results["zscore"] = self.detect_zscore(df)
        self.results["iqr"] = self.detect_iqr(df)
        
        # Compare methods
        for method, res in self.results.items():
            label_col = [c for c in res.columns if c.startswith("anomaly_label")]
            if label_col:
                n_anomalies = (res[label_col[0]] == -1).sum()
                self.comparison[method] = {
                    "anomalies_detected": int(n_anomalies),
                    "anomaly_percentage": round(n_anomalies / len(res) * 100, 2)
                }
        
        return self.results
    
    def get_combined_anomalies(self) -> pd.DataFrame:
        """
        Combine results from all methods.
        An observation is flagged if at least 2 methods agree.
        """
        if not self.results:
            return pd.DataFrame()
        
        base = self.results["isolation_forest"].copy()
        
        label_cols = []
        for method, res in self.results.items():
            col = f"anomaly_label_{method[:2]}" if method == "isolation_forest" else f"anomaly_label_{method}"
            if col in base.columns:
                label_cols.append(col)
        
        if label_cols:
            base["anomaly_votes"] = (base[label_cols] == -1).sum(axis=1)
            base["anomaly_label_consensus"] = np.where(base["anomaly_votes"] >= 2, -1, 1)
        
        return base
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get method comparison table."""
        rows = []
        for method, stats in self.comparison.items():
            rows.append({
                "Method": method.replace("_", " ").title(),
                "Anomalies Detected": stats["anomalies_detected"],
                "Percentage": f"{stats['anomaly_percentage']}%"
            })
        return pd.DataFrame(rows)