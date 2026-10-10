"""
SupplyGuard - Demand Forecasting Module
Multiple regression models with chronological splitting.
"""

import pandas as pd
import numpy as np
import joblib
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

from utils.metrics import regression_metrics
from utils.feature_engineering import FeatureEngineer
from config import MODELS_DIR, METRICS_DIR, RANDOM_SEED, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class DemandForecaster:
    """
    Demand forecasting with multiple models and chronological splitting.
    Prevents data leakage using time-ordered splits.
    """
    
    def __init__(self, target_col: str = "demand", date_col: str = "date"):
        self.target_col = target_col
        self.date_col = date_col
        self.models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.best_model = None
        self.results: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.split_info: Dict = {}
        self.baseline_metrics: Dict = {}
        
    def get_available_models(self) -> Dict[str, Any]:
        """Return dictionary of available models."""
        models = {
            "Linear Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "Random Forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=RANDOM_SEED, n_jobs=-1
                ))
            ]),
            "Gradient Boosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    random_state=RANDOM_SEED
                ))
            ]),
            "LightGBM": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(
                    n_estimators=300, learning_rate=0.03, num_leaves=31,
                    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                    reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1
                ))
            ]),
        }
        
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=RANDOM_SEED, verbosity=0
                ))
            ])
        
        return models
    
    def chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically: 70% train, 15% validation, 15% test.
        This prevents temporal data leakage.
        """
        df = df.sort_values(self.date_col).reset_index(drop=True)
        unique_dates = pd.Index(df[self.date_col].dropna().unique()).sort_values()
        if len(unique_dates) < 3:
            raise ValueError("At least three distinct dates are required for chronological forecasting.")

        train_end = min(max(int(len(unique_dates) * TRAIN_RATIO), 1), len(unique_dates) - 2)
        val_end = min(max(int(len(unique_dates) * (TRAIN_RATIO + VALIDATION_RATIO)), train_end + 1), len(unique_dates) - 1)
        train_cutoff = unique_dates[train_end - 1]
        validation_cutoff = unique_dates[val_end - 1]

        train_df = df[df[self.date_col] <= train_cutoff].copy()
        val_df = df[(df[self.date_col] > train_cutoff) & (df[self.date_col] <= validation_cutoff)].copy()
        test_df = df[df[self.date_col] > validation_cutoff].copy()
        
        self.split_info = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_period": (str(train_df[self.date_col].min()), str(train_df[self.date_col].max())),
            "val_period": (str(val_df[self.date_col].min()), str(val_df[self.date_col].max())),
            "test_period": (str(test_df[self.date_col].min()), str(test_df[self.date_col].max()))
        }
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector."""
        features = df.copy()
        exclude = [self.date_col, self.target_col, "product_id", "warehouse", "supplier_id", "category"]
        if self.target_col == "demand":
            exclude.append("inventory")
        feature_cols = [
            column for column in features.columns
            if column not in exclude
            and pd.api.types.is_numeric_dtype(features[column])
        ]

        X = features[feature_cols].copy()
        if self.target_col == "demand":
            X = X.drop(
                columns=[column for column in ("stockout_label", "demand_forecast") if column in X.columns],
                errors="ignore",
            )
        y = df[self.target_col].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        return X, y

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar predictors and leakage-safe history features for demand."""
        engineer = FeatureEngineer(df, date_col=self.date_col, target_col=self.target_col)
        engineer.create_temporal_features()
        if self.target_col == "demand":
            engineer.create_demand_features()
        return engineer.df
    
    def compute_baseline(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Compute naive and moving average baselines."""
        y_test = test_df[self.target_col].values
        
        # Naive forecast: use previous value
        if "lag_1" in test_df.columns:
            naive_pred = test_df["lag_1"].fillna(train_df[self.target_col].mean()).values
            self.baseline_metrics["Naive (lag_1)"] = regression_metrics(y_test, naive_pred)
        
        # Moving average
        if "rolling_mean_7" in test_df.columns:
            ma_pred = test_df["rolling_mean_7"].fillna(train_df[self.target_col].mean()).values
            self.baseline_metrics["Moving Average (7)"] = regression_metrics(y_test, ma_pred)

        if self.target_col == "demand" and "demand_forecast" in test_df.columns:
            provided_forecast = test_df["demand_forecast"].fillna(train_df[self.target_col].mean()).values
            self.baseline_metrics["Dataset-provided demand forecast"] = regression_metrics(
                y_test, provided_forecast, y_train=train_df[self.target_col]
            )

    def tune_lightgbm(self, model, X: pd.DataFrame, y: pd.Series, dates: pd.Series):
        """Tune LightGBM with expanding-window folds made from whole dates."""
        unique_dates = pd.Index(pd.to_datetime(dates, errors="coerce").dropna().unique()).sort_values()
        if len(unique_dates) < 4:
            return model, {}

        parameter_candidates = [
            {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 15},
            {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31},
            {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31},
            {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 63},
        ]
        splitter = TimeSeriesSplit(n_splits=min(3, len(unique_dates) - 1))
        fold_indices = list(splitter.split(unique_dates))
        date_values = pd.to_datetime(dates, errors="coerce")
        best_model = model
        best_rmse = np.inf
        best_params = {}

        for params in parameter_candidates:
            fold_errors = []
            for train_date_indices, validation_date_indices in fold_indices:
                fold_train_dates = unique_dates[train_date_indices]
                fold_validation_dates = unique_dates[validation_date_indices]
                train_rows = date_values.isin(fold_train_dates)
                validation_rows = date_values.isin(fold_validation_dates)
                candidate = clone(model).set_params(**{
                    f"model__{key}": value for key, value in params.items()
                })
                candidate.fit(X.loc[train_rows], y.loc[train_rows])
                predictions = candidate.predict(X.loc[validation_rows])
                fold_errors.append(np.sqrt(np.mean((y.loc[validation_rows].to_numpy() - predictions) ** 2)))

            mean_rmse = float(np.mean(fold_errors))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = params
                best_model = clone(model).set_params(**{
                    f"model__{key}": value for key, value in params.items()
                })

        return best_model, {
            "cv_rmse": round(best_rmse, 4),
            "cv_folds": len(fold_indices),
            "params": best_params,
        }
    
    def train_all(self, df: pd.DataFrame, progress_callback=None) -> Dict[str, Dict]:
        """
        Train all models and return results.
        Uses chronological splitting.
        """
        df = self.engineer_features(df)
        train_df, val_df, test_df = self.chronological_split(df)
        
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        X_test, y_test = self.prepare_features(test_df)
        
        self.feature_names = X_train.columns.tolist()
        
        # Drop rows with missing targets in each split.
        train_mask = y_train.notna()
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        validation_mask = y_val.notna()
        X_val, y_val = X_val[validation_mask], y_val[validation_mask]
        test_mask = y_test.notna()
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        
        # Compute baselines
        self.compute_baseline(train_df, test_df)
        
        available_models = self.get_available_models()
        results = {}
        
        for i, (name, model) in enumerate(available_models.items()):
            if progress_callback:
                progress_callback(i / len(available_models), f"Training {name}...")
            
            try:
                tuning_result = {}
                if name == "LightGBM":
                    train_dates = train_df.loc[train_mask, self.date_col]
                    model, tuning_result = self.tune_lightgbm(model, X_train, y_train, train_dates)

                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                validation_pred = model.predict(X_val)
                
                results[name] = {
                    "model": model,
                    "validation_metrics": regression_metrics(y_val, validation_pred, y_train=y_train),
                    "training_time": round(train_time, 3),
                    "validation_pred": validation_pred,
                    **tuning_result,
                }
                
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "validation_metrics": {},
                    "test_metrics": {},
                    "training_time": 0
                }

        # Select on validation data, then evaluate refitted candidates on the untouched test period.
        valid_results = {k: v for k, v in results.items() 
                        if "validation_metrics" in v and v["validation_metrics"].get("RMSE") is not None
                        and not np.isnan(v["validation_metrics"].get("RMSE", np.nan))}
        
        if valid_results:
            self.best_model_name = min(valid_results.keys(),
                                       key=lambda k: valid_results[k]["validation_metrics"]["RMSE"])

            X_fit = pd.concat([X_train, X_val])
            y_fit = pd.concat([y_train, y_val])
            for name, result in valid_results.items():
                try:
                    model = result["model"]
                    model.fit(X_fit, y_fit)
                    y_test_pred = model.predict(X_test)
                    result["test_metrics"] = regression_metrics(y_test, y_test_pred, y_train=y_fit)
                    result["y_test"] = y_test.values
                    result["y_pred"] = y_test_pred
                    result["test_dates"] = test_df.loc[test_mask, self.date_col].values
                    self.models[name] = model
                except Exception as e:
                    result["error"] = str(e)
                    result["test_metrics"] = {}

            self.best_model = self.models.get(self.best_model_name)

        self.results = results
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        X, _ = self.prepare_features(df)
        return self.best_model.predict(X)
    
    def forecast_future(self, df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        """
        Generate recursive forecasts using each predicted period as history.
        """
        if self.best_model is None:
            raise ValueError("No trained model available.")

        history = df.sort_values(self.date_col).reset_index(drop=True).copy()
        last_date = pd.to_datetime(history[self.date_col].max())
        if self.target_col == "demand":
            group_cols = [column for column in ("product_id", "warehouse") if column in history.columns]
        else:
            group_cols = []
            for raw_column, encoded_column in (
                ("product_id", "product_id_encoded"), ("commodity", "commodity_encoded"),
                ("food", "food_encoded"), ("market", "market_encoded"),
                ("country", "country_encoded"), ("warehouse", "warehouse_encoded"),
            ):
                selected = raw_column if raw_column in history.columns else encoded_column
                if selected in history.columns:
                    group_cols.append(selected)

        forecasts = []
        unique_dates = pd.DatetimeIndex(pd.to_datetime(history[self.date_col], errors="coerce").dropna().unique()).sort_values()
        inferred_frequency = pd.infer_freq(unique_dates) if len(unique_dates) >= 3 else None
        for h in range(1, horizon + 1):
            if inferred_frequency:
                future_date = last_date + pd.tseries.frequencies.to_offset(inferred_frequency) * h
            else:
                future_date = last_date + pd.Timedelta(days=h)

            if group_cols:
                latest = history.sort_values(self.date_col).groupby(group_cols, dropna=False).tail(1).copy()
            else:
                latest = history.tail(1).copy()
            future_rows = latest.copy()
            future_rows[self.date_col] = future_date
            future_rows[self.target_col] = np.nan
            combined = pd.concat([history, future_rows], ignore_index=True)
            engineered = self.engineer_features(combined)
            future_features = engineered.tail(len(future_rows))
            X_future, _ = self.prepare_features(future_features)
            preds = self.best_model.predict(X_future)
            
            future_rows[self.target_col] = np.maximum(0, preds)
            for i, (_, row) in enumerate(future_rows.iterrows()):
                record = {
                    "forecast_date": future_date,
                    "horizon": h,
                    f"predicted_{self.target_col}": max(0, preds[i])
                }
                for col in group_cols:
                    record[col] = row[col]
                forecasts.append(record)

            history = pd.concat([history, future_rows], ignore_index=True)
        
        return pd.DataFrame(forecasts)
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get model comparison table."""
        columns = [
            "Model", "Validation RMSE", "Test RMSE", "MAE", "R2", "MAPE",
            "WAPE", "sMAPE", "MASE", "Training Time (s)",
        ]
        rows = []
        for name, result in self.results.items():
            if "error" in result:
                continue
            tm = result["test_metrics"]
            validation_metrics = result.get("validation_metrics", {})
            rows.append({
                "Model": name,
                "Validation RMSE": validation_metrics.get("RMSE", np.nan),
                "RMSE": tm.get("RMSE", np.nan),
                "Test RMSE": tm.get("RMSE", np.nan),
                "MAE": tm.get("MAE", np.nan),
                "R2": tm.get("R2", np.nan),
                "MAPE": tm.get("MAPE", np.nan),
                "WAPE": tm.get("WAPE", np.nan),
                "sMAPE": tm.get("sMAPE", np.nan),
                "MASE": tm.get("MASE", np.nan),
                "Training Time (s)": result["training_time"]
            })
        
        return pd.DataFrame(rows, columns=columns).sort_values("Validation RMSE").reset_index(drop=True)
    
    def save(self, path_prefix: str = None):
        """Save the best model and metadata."""
        if self.best_model is None:
            return
        
        prefix = path_prefix or str(MODELS_DIR / "demand_model")
        
        joblib.dump(self.best_model, f"{prefix}.pkl")
        
        metadata = {
            "model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "target_col": self.target_col,
            "date_col": self.date_col,
            "split_info": self.split_info,
            "test_metrics": self.results.get(self.best_model_name, {}).get("test_metrics", {}),
            "training_timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED
        }
        
        import json
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path_prefix: str = None):
        """Load a saved model."""
        prefix = path_prefix or str(MODELS_DIR / "demand_model")
        model = joblib.load(f"{prefix}.pkl")
        
        import json
        with open(f"{prefix}_metadata.json", "r") as f:
            metadata = json.load(f)
        
        forecaster = cls(target_col=metadata["target_col"], date_col=metadata["date_col"])
        forecaster.best_model = model
        forecaster.best_model_name = metadata["model_name"]
        forecaster.feature_names = metadata["feature_names"]
        forecaster.split_info = metadata.get("split_info", {})
        
        return forecaster