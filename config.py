"""
SupplyGuard - Configuration Module
Central configuration for the entire application.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR,
                  MODELS_DIR, METRICS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Data configuration
DEFAULT_DATE_COLUMN = "date"
DEFAULT_TARGET_COLUMN = "demand"
DEFAULT_PRODUCT_COLUMN = "product_id"
DEFAULT_WAREHOUSE_COLUMN = "warehouse"
DEFAULT_SUPPLIER_COLUMN = "supplier_id"

# Expected columns (not mandatory - system adapts)
EXPECTED_COLUMNS = [
    "date", "product_id", "category", "warehouse", "supplier_id",
    "price", "promotion", "holiday", "inventory", "lead_time",
    "competitor_price", "demand"
]

# Risk thresholds (configurable)
RISK_THRESHOLDS = {
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 1.00
}

# Stock-out risk levels
STOCKOUT_RISK_LEVELS = {
    "LOW": (0.0, 0.25),
    "MEDIUM": (0.25, 0.50),
    "HIGH": (0.50, 0.75),
    "CRITICAL": (0.75, 1.00)
}

# Supplier risk levels
SUPPLIER_RISK_LEVELS = {
    "LOW": (0.0, 0.25),
    "MEDIUM": (0.25, 0.50),
    "HIGH": (0.50, 0.75),
    "CRITICAL": (0.75, 1.00)
}

# Inventory status thresholds
INVENTORY_STATUS = {
    "CRITICAL_STOCK": 0.25,
    "LOW_STOCK": 0.50,
    "HEALTHY": 0.80,
    "OVERSTOCK": 1.50
}

# Time-series split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Forecast horizons (days)
FORECAST_HORIZONS = [1, 7, 14, 30]

# Model hyperparameters defaults
DEFAULT_HYPERPARAMS = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": RANDOM_SEED
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": RANDOM_SEED
    }
}

# Sustainability weights (relative indicators - configurable)
SUSTAINABILITY_WEIGHTS = {
    "inventory_waste_risk": 0.25,
    "transportation_burden": 0.25,
    "emergency_replenishment": 0.25,
    "redistribution_efficiency": 0.25
}

# Application settings
APP_TITLE = "SupplyGuard"
APP_SUBTITLE = "Explainable AI for Risk-Aware & Sustainable Supply Chain Management"
APP_VERSION = "1.0.0"
