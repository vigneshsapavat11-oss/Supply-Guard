"""
SupplyGuard - Data Preprocessing Module
Robust preprocessing pipeline with schema detection and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st
from config import DEFAULT_DATE_COLUMN, DEFAULT_TARGET_COLUMN


class DataPreprocessor:
    """
    Robust data preprocessing pipeline for supply chain data.
    Handles schema detection, missing values, duplicates, and encoding.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.report = {
            "rows_before": len(df),
            "rows_after": 0,
            "missing_values": {},
            "duplicates_removed": 0,
            "numerical_features": [],
            "categorical_features": [],
            "date_range": None,
            "target_variable": None,
            "preprocessing_steps": []
        }
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.date_column: Optional[str] = None
        self.target_column: Optional[str] = None

    def normalize_columns(self) -> pd.DataFrame:
        """Map common supply-chain CSV header variants to the app's standard schema."""
        aliases = {
            "date": ("date", "orderdate", "transactiondate", "invoicedate", "timestamp"),
            "product_id": ("productid", "skuid", "sku", "itemid", "itemcode"),
            "warehouse": ("warehouse", "warehouseid", "store", "storeid"),
            "supplier_id": ("supplierid", "supplier", "suppliername", "vendorid", "vendorname"),
            "demand": ("demand", "unitssold", "sales", "quantity", "salesquantity", "unitsordered"),
            "inventory": ("inventory", "inventorylevel", "stocklevel", "stockonhand", "availableinventory"),
            "lead_time": ("leadtime", "supplierleadtime", "supplierleadtimedays", "leadtimedays"),
            "price": ("price", "unitprice", "sellingprice"),
            "promotion": ("promotion", "promotionflag", "promoflag"),
            "holiday": ("holiday", "holidayflag"),
            "stockout_label": ("stockoutlabel", "stockoutflag", "stockout"),
            "demand_forecast": ("demandforecast", "salesforecast"),
        }
        normalized_columns = {
            "".join(character.lower() for character in str(column) if character.isalnum()): column
            for column in self.df.columns
        }
        rename = {}
        for canonical, variants in aliases.items():
            if canonical in self.df.columns:
                continue
            source = next(
                (normalized_columns[variant] for variant in variants if variant in normalized_columns),
                None,
            )
            if source is not None:
                rename[source] = canonical

        if rename:
            self.df = self.df.rename(columns=rename)
            renamed = ", ".join(f"{source} -> {target}" for source, target in rename.items())
            self.report["preprocessing_steps"].append(f"Normalized supply-chain columns: {renamed}")
        return self.df
        
    def detect_schema(self) -> Dict[str, Any]:
        """Detect column types and identify key columns."""
        schema = {
            "numerical": [],
            "categorical": [],
            "datetime": [],
            "boolean": []
        }
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                schema["numerical"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                schema["datetime"].append(col)
            elif self.df[col].dtype == bool:
                schema["boolean"].append(col)
            else:
                schema["categorical"].append(col)
        
        # Try to detect date columns from strings
        for col in schema["categorical"][:]:
            sample = self.df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() >= 0.8:
                        schema["datetime"].append(col)
                        schema["categorical"].remove(col)
                except (ValueError, TypeError):
                    pass
        
        return schema
    
    def parse_dates(self) -> bool:
        """Parse date columns with validation."""
        schema = self.detect_schema()
        date_cols = schema.get("datetime", [])
        
        # Look for common date column names
        date_candidates = ["date", "order_date", "transaction_date", "timestamp", "ds"]
        
        for col in self.df.columns:
            if col.lower() in date_candidates:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    invalid_count = self.df[col].isna().sum()
                    if invalid_count > 0:
                        st.warning(f"⚠️ {invalid_count} invalid dates found in '{col}'. These rows will be handled.")
                    self.date_column = col
                    self.report["preprocessing_steps"].append(f"Parsed date column: {col}")
                    return True
                except Exception as e:
                    st.error(f"❌ Failed to parse dates in '{col}': {e}")
        
        if date_cols:
            self.date_column = date_cols[0]
            try:
                self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors="coerce")
                self.report["preprocessing_steps"].append(f"Parsed date column: {self.date_column}")
                return True
            except Exception:
                pass
        
        st.info("No date column detected. Time-based forecasting and trend analysis require a date column.")
        return False
    
    def identify_target(self, target_col: Optional[str] = None) -> str:
        """Identify or set the target variable."""
        if target_col and target_col in self.df.columns:
            self.target_column = target_col
        elif DEFAULT_TARGET_COLUMN in self.df.columns:
            self.target_column = DEFAULT_TARGET_COLUMN
        else:
            # Look for demand-related columns
            candidates = ["demand", "sales", "quantity", "units_sold", "target", "y"]
            for col in self.df.columns:
                if col.lower() in candidates:
                    self.target_column = col
                    break
        
        if self.target_column:
            self.report["target_variable"] = self.target_column
            self.report["preprocessing_steps"].append(f"Target variable: {self.target_column}")
        else:
            st.info(
                "No demand-like target detected. Supply-chain forecasting requires a numeric demand, sales, quantity, or price column."
            )
        
        return self.target_column or ""
    
    def handle_missing_values(self, strategy: str = "auto") -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        missing_before = self.df.isnull().sum()
        self.report["missing_values"] = missing_before[missing_before > 0].to_dict()
        
        if missing_before.sum() == 0:
            self.report["preprocessing_steps"].append("No missing values found")
            return self.df
        
        # Numerical columns: median imputation
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        # Categorical columns: mode imputation
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col] = self.df[col].fillna(mode_val[0])
                else:
                    self.df[col] = self.df[col].fillna("Unknown")
        
        # Date columns: drop rows with invalid dates
        if self.date_column and self.df[self.date_column].isnull().any():
            before = len(self.df)
            self.df = self.df.dropna(subset=[self.date_column])
            dropped = before - len(self.df)
            if dropped > 0:
                self.report["preprocessing_steps"].append(f"Dropped {dropped} rows with invalid dates")
        
        self.report["preprocessing_steps"].append("Handled missing values (median for numerical, mode for categorical)")
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        self.report["duplicates_removed"] = before - after
        if before - after > 0:
            self.report["preprocessing_steps"].append(f"Removed {before - after} duplicate rows")
        return self.df
    
    def detect_outliers(self, columns: Optional[List[str]] = None, method: str = "iqr") -> pd.DataFrame:
        """
        Detect outliers but do not remove them automatically.
        Returns a report of outlier counts.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_report = {}
        for col in columns:
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            elif method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = (z_scores > 3).sum()
            else:
                outliers = 0
            outlier_report[col] = int(outliers)
        
        self.report["outliers"] = outlier_report
        return self.df
    
    def encode_categoricals(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding."""
        if columns is None:
            columns = self.df.select_dtypes(include=["object"]).columns.tolist()
            # Exclude date column
            if self.date_column in columns:
                columns.remove(self.date_column)
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + "_encoded"] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                self.report["categorical_features"].append(col)
        
        self.report["preprocessing_steps"].append(f"Encoded {len(columns)} categorical columns")
        return self.df
    
    def get_numerical_features(self) -> List[str]:
        """Get list of numerical feature columns (excluding target)."""
        numerical = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical:
            numerical.remove(self.target_column)
        # Remove encoded columns from the list if original exists
        return [c for c in numerical if not c.endswith("_encoded") or c.replace("_encoded", "") not in self.df.columns]
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature columns."""
        return self.df.select_dtypes(include=["object"]).columns.tolist()
    
    def run_pipeline(self, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete preprocessing pipeline."""
        self.normalize_columns()
        self.parse_dates()
        self.identify_target(target_col)
        self.remove_duplicates()
        self.handle_missing_values()
        self.detect_outliers()
        self.encode_categoricals()
        
        self.report["rows_after"] = len(self.df)
        self.report["numerical_features"] = self.get_numerical_features()
        self.report["categorical_features"] = self.get_categorical_features()
        
        if self.date_column:
            self.report["date_range"] = (
                str(self.df[self.date_column].min()),
                str(self.df[self.date_column].max())
            )
        
        return self.df, self.report


@st.cache_data(show_spinner=False)
def load_and_preprocess(file_path_or_buffer, target_col: Optional[str] = None):
    """Cached loading and preprocessing."""
    if isinstance(file_path_or_buffer, pd.DataFrame):
        df = file_path_or_buffer.copy()
    elif isinstance(file_path_or_buffer, str):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_csv(file_path_or_buffer)
    
    preprocessor = DataPreprocessor(df)
    processed_df, report = preprocessor.run_pipeline(target_col)
    return processed_df, report, preprocessor