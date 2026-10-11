"""SupplyGuard - About Page"""

import streamlit as st
from utils.ui_theme import apply_enterprise_theme
from config import APP_TITLE, APP_SUBTITLE, APP_VERSION

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
apply_enterprise_theme()

st.markdown("# ℹ️ About SupplyGuard")

st.markdown(f"""
## {APP_TITLE}

**{APP_SUBTITLE}**

Version: {APP_VERSION}
""")

st.markdown("---")

st.markdown("""
### 🎯 Project Overview

SupplyGuard is a research-oriented **Explainable AI Framework** for 
**Risk-Aware and Sustainable Supply Chain Management**. It integrates multiple 
machine learning and explainability techniques into a unified decision-support system.

### ✨ Key Features

| Module | Description |
|--------|-------------|
| **Demand Forecasting** | Multi-model demand prediction with proper regression metrics |
| **Stock-Out Early Warning** | Classification-based inventory depletion risk |
| **Supplier Risk Prediction** | Operational feature-based supplier risk assessment |
| **Anomaly Detection** | Isolation Forest + statistical methods for demand shocks |
| **Warehouse Optimization** | Multi-warehouse redistribution recommendations |
| **Explainable AI** | SHAP-based global and local explanations |
| **Sustainability** | Relative sustainability indicators |
| **What-If Simulation** | Scenario analysis for decision support |
| **Research Experiments** | Reproducible experiment logging |

### 🏗️ System Architecture

SupplyGuard follows a modular pipeline:

1. **Data Ingestion** — CSV upload or bundled sample dataset
2. **Preprocessing** — Cleaning, type coercion, feature engineering
3. **Model Training** — Multiple ML models with time-aware validation
4. **Explainability** — SHAP values for global and local interpretation
5. **Decision Support** — Risk scoring, redistribution, what-if analysis

---

*Built with Streamlit, scikit-learn, XGBoost, LightGBM, and SHAP.*
""")
