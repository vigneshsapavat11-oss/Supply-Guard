"""
SupplyGuard - Main Application Entry Point
Explainable AI Framework for Risk-Aware & Sustainable Supply Chain Management
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION, DATA_DIR,
    TEAM_MEMBERS, PROJECT_GUIDE, GUIDE_DESIGNATION, GUIDE_DEPARTMENT
)
from utils.data_preprocessing import load_and_preprocess
from utils.ui_theme import apply_enterprise_theme
from models.model_manager import get_model_manager

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_enterprise_theme()


def init_session_state():
    """Initialize session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "preprocessing_report" not in st.session_state:
        st.session_state.preprocessing_report = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False
    if "training_status" not in st.session_state:
        st.session_state.training_status = {}
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "experiment_log" not in st.session_state:
        st.session_state.experiment_log = []


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    with st.sidebar:
        st.caption(f"v{APP_VERSION}")
        st.markdown("---")
        
        # Dataset selection
        st.markdown("### 📁 Dataset")
        data_source = st.radio(
            "Data Source",
            ["Use Sample Dataset", "Upload CSV"],
            index=0,
            key="data_source_selection"
        )
        
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        st.markdown("---")
        st.markdown("### ℹ️ Navigation")
        st.caption("Use the pages menu above to navigate between modules.")
        
        st.markdown("---")
        st.markdown("### 👥 Team")
        for member in TEAM_MEMBERS:
            st.caption(f"• {member}")
        st.caption(f"\n**Guide:** {PROJECT_GUIDE}")
        
        return data_source, uploaded_file


def load_data(data_source: str, uploaded_file):
    """Load and preprocess data."""
    try:
        if data_source == "Use Sample Dataset":
            sample_path = DATA_DIR / "sample_supply_chain.csv"
            if not sample_path.exists():
                st.error("⚠️ Sample dataset not found. Please run `python data/generate_sample_data.py` first.")
                return False, None, None, None
            df, report, preprocessor = load_and_preprocess(str(sample_path))
        else:
            if uploaded_file is None:
                st.warning("Choose a CSV file before loading the dataset.")
                return False, None, None, None
            df, report, preprocessor = load_and_preprocess(uploaded_file)

        if df.empty:
            st.error("The selected dataset contains no usable rows.")
            return False, None, None, None
        
        st.session_state.processed_df = df
        st.session_state.preprocessing_report = report
        st.session_state.preprocessor = preprocessor
        st.session_state.data_loaded = True
        
        return True, df, report, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return False, None, None, None


def train_models(df):
    """Train available models and report each model's outcome."""
    manager = get_model_manager()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress, message):
        progress_bar.progress(min(max(progress, 0.0), 1.0))
        status_text.text(message)

    with st.spinner("Training models..."):
        results = manager.train_all(df, progress_callback=update_progress)

    status = manager.get_status()
    st.session_state.training_status = status
    st.session_state.models_trained = manager.is_trained()

    status_rows = [
        {"Model": name.replace("_", " ").title(), "Status": model_status}
        for name, model_status in status.items()
    ]
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)

    completed = sum(value == "complete" or value.startswith("complete (") for value in status.values())
    unavailable = sum(value.startswith("unavailable:") for value in status.values())
    failed = len(status) - completed - unavailable
    if unavailable or failed:
        st.warning(
            f"{completed} of {len(status)} modules completed; {unavailable} unavailable and {failed} failed for this dataset. "
            "See each module's status for the required columns."
        )
    else:
        st.success(f"All {completed} model modules are ready. Open a page from the navigation menu to explore the results.")

    return results


def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.markdown(f'<div class="main-header">🛡️ {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{APP_SUBTITLE}</div>', unsafe_allow_html=True)
    
    # Sidebar
    data_source, uploaded_file = render_sidebar()
    
    # Data loading
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Welcome to SupplyGuard")
        st.markdown("""
        SupplyGuard is a research-oriented **Explainable AI Framework** for 
        **Risk-Aware and Sustainable Supply Chain Management**. It integrates:
        
        - 📈 **Demand Forecasting** with multiple ML models
        - ⚠️ **Stock-Out Early Warning** system
        - 🏭 **Supplier Risk Prediction**
        - 🔍 **Demand Anomaly Detection**
        - 📦 **Multi-Warehouse Redistribution**
        - 🧠 **SHAP Explainable AI**
        - 🌱 **Sustainability Indicators**
        - 🔄 **What-If Simulation**
        """)
    
    with col2:
        st.markdown("#### Quick Start")
        if st.button("Load Dataset", use_container_width=True):
            with st.spinner("Loading dataset..."):
                success, df, report, preprocessor = load_data(data_source, uploaded_file)
                if success:
                    st.success(f"✅ Dataset loaded: {len(df)} rows")

        if st.button("Load & Train", use_container_width=True, type="primary"):
            success, df, _, _ = load_data(data_source, uploaded_file)
            if success:
                st.success(f"Dataset loaded: {len(df):,} rows")
                train_models(df)

        if st.session_state.data_loaded and st.button("Train Models", use_container_width=True):
            train_models(st.session_state.processed_df)
    
    # Data status
    if st.session_state.data_loaded:
        df = st.session_state.processed_df
        report = st.session_state.preprocessing_report
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Status")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{report['rows_after']:,}")
        col2.metric("Columns", f"{len(df.columns)}")
        col3.metric("Numerical Features", len(report.get("numerical_features", [])))
        col4.metric("Categorical Features", len(report.get("categorical_features", [])))
        
        with st.expander("📋 View Preprocessing Report"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dataset Summary**")
                st.write(f"- Rows before: {report['rows_before']}")
                st.write(f"- Rows after: {report['rows_after']}")
                st.write(f"- Duplicates removed: {report['duplicates_removed']}")
                st.write(f"- Target variable: {report.get('target_variable', 'N/A')}")
            with col2:
                st.markdown("**Preprocessing Steps**")
                for step in report.get("preprocessing_steps", []):
                    st.write(f"• {step}")
            
            if report.get("date_range"):
                st.markdown(f"**Date Range:** {report['date_range'][0]} to {report['date_range'][1]}")
        
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(100), use_container_width=True)

        if st.session_state.training_status:
            with st.expander("Model training status"):
                status_rows = [
                    {"Model": name.replace("_", " ").title(), "Status": status}
                    for name, status in st.session_state.training_status.items()
                ]
                st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
        
        # Download processed data
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Processed Dataset",
            csv,
            "supplyguard_processed.csv",
            "text/csv"
        )
    
    else:
        st.markdown("---")
        st.info("👈 Select a dataset source in the sidebar and click **Load & Train** to begin.")
        
        # Show sample dataset info
        st.markdown("### 📋 Expected Dataset Format")
        st.markdown("""
        The system works best with data containing:
        
        | Column | Description | Required |
        |--------|-------------|----------|
        | `date` | Transaction date | Recommended |
        | `product_id` | Product identifier | Recommended |
        | `warehouse` | Warehouse identifier | Recommended |
        | `supplier_id` | Supplier identifier | Recommended |
        | `demand` | Target variable (demand) | Required for forecasting |
        | `inventory` | Current inventory level | Required for stock-out |
        | `price` | Product price | Optional |
        | `promotion` | Promotion flag (0/1) | Optional |
        | `holiday` | Holiday flag (0/1) | Optional |
        | `lead_time` | Supplier lead time | Optional |
        | `competitor_price` | Competitor price | Optional |
        
        The system adapts to available columns. Missing optional columns are handled gracefully.
        """)


if __name__ == "__main__":
    main()