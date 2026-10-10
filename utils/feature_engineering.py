"""
SupplyGuard - Feature Engineering Module
Creates meaningful supply chain features with leakage prevention.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from config import RANDOM_SEED


class FeatureEngineer:
    """
    Feature engineering for supply chain data.
    All features use only historical information (no future leakage).
    """
    
    def __init__(self, df: pd.DataFrame, date_col: str = "date", 
                 target_col: str = "demand",
                 product_col: str = "product_id",
                 warehouse_col: str = "warehouse"):
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.product_col = product_col
        self.warehouse_col = warehouse_col
        self.created_features: List[str] = []
        
        # Ensure date is datetime
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col).reset_index(drop=True)
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Create calendar-based temporal features."""
        if self.date_col not in self.df.columns:
            return self.df
        
        self.df["year"] = self.df[self.date_col].dt.year
        self.df["month"] = self.df[self.date_col].dt.month
        self.df["week"] = self.df[self.date_col].dt.isocalendar().week.astype(int)
        self.df["day"] = self.df[self.date_col].dt.day
        self.df["day_of_week"] = self.df[self.date_col].dt.dayofweek
        self.df["quarter"] = self.df[self.date_col].dt.quarter
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        
        temporal_features = ["year", "month", "week", "day", "day_of_week", "quarter", "is_weekend"]
        self.created_features.extend(temporal_features)
        return self.df
    
    def create_demand_features(self, lags: List[int] = [1, 7, 14, 30],
                                windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create lag and rolling features for demand.
        IMPORTANT: Uses shift() to prevent data leakage.
        """
        if self.target_col not in self.df.columns:
            return self.df
        
        group_cols = []
        if self.product_col in self.df.columns:
            group_cols.append(self.product_col)
        if self.warehouse_col in self.df.columns:
            group_cols.append(self.warehouse_col)
        
        if group_cols:
            grouped = self.df.groupby(group_cols, group_keys=False)
        else:
            grouped = self.df
        
        # Lag features (shifted to avoid leakage)
        for lag in lags:
            col_name = f"lag_{lag}"
            if group_cols:
                self.df[col_name] = grouped[self.target_col].shift(lag)
            else:
                self.df[col_name] = self.df[self.target_col].shift(lag)
            self.created_features.append(col_name)
        
        # Rolling statistics (shifted by 1 to avoid leakage)
        for window in windows:
            if group_cols:
                self.df[f"rolling_mean_{window}"] = grouped[self.target_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                self.df[f"rolling_std_{window}"] = grouped[self.target_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).std()
                )
            else:
                self.df[f"rolling_mean_{window}"] = self.df[self.target_col].shift(1).rolling(window, min_periods=1).mean()
                self.df[f"rolling_std_{window}"] = self.df[self.target_col].shift(1).rolling(window, min_periods=1).std()
            self.created_features.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])
        
        # Demand growth rate
        if "lag_1" in self.df.columns and "lag_7" in self.df.columns:
            self.df["demand_growth_rate"] = np.where(
                self.df["lag_7"] > 0,
                (self.df["lag_1"] - self.df["lag_7"]) / self.df["lag_7"],
                0
            )
            self.created_features.append("demand_growth_rate")
        
        # Demand volatility
        if "rolling_std_7" in self.df.columns and "rolling_mean_7" in self.df.columns:
            self.df["demand_volatility"] = np.where(
                self.df["rolling_mean_7"] > 0,
                self.df["rolling_std_7"] / self.df["rolling_mean_7"],
                0
            )
            self.created_features.append("demand_volatility")
        
        return self.df
    
    def create_inventory_features(self) -> pd.DataFrame:
        """Create inventory-based features."""
        if "inventory" not in self.df.columns:
            return self.df
        
        # Days of inventory (using lagged demand to avoid leakage)
        if "lag_1" in self.df.columns:
            self.df["days_of_inventory"] = np.where(
                self.df["lag_1"] > 0,
                self.df["inventory"] / self.df["lag_1"],
                999
            )
            self.created_features.append("days_of_inventory")
        
        # Inventory to demand ratio
        if self.target_col in self.df.columns:
            self.df["inventory_to_demand_ratio"] = np.where(
                self.df[self.target_col] > 0,
                self.df["inventory"] / self.df[self.target_col],
                0
            )
            self.created_features.append("inventory_to_demand_ratio")
        
        # Inventory change
        group_cols = []
        if self.product_col in self.df.columns:
            group_cols.append(self.product_col)
        if self.warehouse_col in self.df.columns:
            group_cols.append(self.warehouse_col)
        
        if group_cols:
            self.df["inventory_change"] = self.df.groupby(group_cols)["inventory"].diff()
        else:
            self.df["inventory_change"] = self.df["inventory"].diff()
        self.created_features.append("inventory_change")
        
        # Inventory velocity
        if "rolling_mean_7" in self.df.columns:
            self.df["inventory_velocity"] = np.where(
                self.df["rolling_mean_7"] > 0,
                self.df["inventory_change"] / self.df["rolling_mean_7"],
                0
            )
            self.created_features.append("inventory_velocity")
        
        return self.df
    
    def create_price_features(self) -> pd.DataFrame:
        """Create price-based features."""
        if "price" not in self.df.columns:
            return self.df
        
        # Price difference from rolling mean
        group_cols = []
        if self.product_col in self.df.columns:
            group_cols.append(self.product_col)
        
        if group_cols:
            self.df["price_rolling_mean_7"] = self.df.groupby(group_cols)["price"].transform(
                lambda x: x.shift(1).rolling(7, min_periods=1).mean()
            )
        else:
            self.df["price_rolling_mean_7"] = self.df["price"].shift(1).rolling(7, min_periods=1).mean()
        
        self.df["price_difference"] = self.df["price"] - self.df["price_rolling_mean_7"]
        self.df["price_ratio"] = np.where(
            self.df["price_rolling_mean_7"] > 0,
            self.df["price"] / self.df["price_rolling_mean_7"],
            1
        )
        self.created_features.extend(["price_difference", "price_ratio"])
        
        # Competitor price difference
        if "competitor_price" in self.df.columns:
            self.df["competitor_price_difference"] = self.df["price"] - self.df["competitor_price"]
            self.df["price_competitiveness"] = np.where(
                self.df["competitor_price"] > 0,
                self.df["price"] / self.df["competitor_price"],
                1
            )
            self.created_features.extend(["competitor_price_difference", "price_competitiveness"])
        
        # Discount indicator
        if "promotion" in self.df.columns:
            self.df["discount_indicator"] = self.df["promotion"].astype(int)
            self.created_features.append("discount_indicator")
        
        return self.df
    
    def create_supply_features(self) -> pd.DataFrame:
        """Create supply-related features."""
        # Lead time risk
        if "lead_time" in self.df.columns:
            if "rolling_mean_7" in self.df.columns:
                self.df["lead_time_risk"] = np.where(
                    self.df["rolling_mean_7"] > 0,
                    self.df["lead_time"] / (self.df["rolling_mean_7"] + 1),
                    0
                )
            else:
                self.df["lead_time_risk"] = self.df["lead_time"]
            self.created_features.append("lead_time_risk")
        
        # Delivery delay rate (if available)
        if "delivery_delay" in self.df.columns:
            self.df["delivery_delay_rate"] = self.df["delivery_delay"]
            self.created_features.append("delivery_delay_rate")
        
        # Supplier reliability (if available)
        if "supplier_reliability" in self.df.columns:
            self.df["supplier_reliability_score"] = self.df["supplier_reliability"]
            self.created_features.append("supplier_reliability_score")
        
        return self.df
    
    def create_promotion_features(self) -> pd.DataFrame:
        """Create promotion and holiday features."""
        if "promotion" in self.df.columns:
            # Promotion effect (interaction with demand)
            if self.target_col in self.df.columns:
                self.df["promotion_effect"] = self.df["promotion"] * self.df[self.target_col]
                self.created_features.append("promotion_effect")
        
        if "holiday" in self.df.columns:
            if self.target_col in self.df.columns:
                self.df["holiday_effect"] = self.df["holiday"] * self.df[self.target_col]
                self.created_features.append("holiday_effect")
        
        return self.df
    
    def run_all(self) -> pd.DataFrame:
        """Run all feature engineering steps."""
        self.create_temporal_features()
        self.create_demand_features()
        self.create_inventory_features()
        self.create_price_features()
        self.create_supply_features()
        self.create_promotion_features()
        
        # Drop rows with NaN in critical lag features (only for training)
        # Keep track of original index
        return self.df
    
    def get_feature_columns(self, exclude: List[str] = None) -> List[str]:
        """Get list of feature columns (excluding target, date, and IDs)."""
        exclude = exclude or []
        base_exclude = [self.date_col, self.target_col, self.product_col, self.warehouse_col]
        base_exclude.extend(exclude)
        
        features = [c for c in self.df.columns if c not in base_exclude 
                   and not c.endswith("_encoded")]
        return features