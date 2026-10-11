"""
SupplyGuard - Main Application Entry Point
Explainable AI Framework for Risk-Aware & Sustainable Supply Chain Management

Author: vigneshsapavat11-oss
Version: See config.APP_VERSION
"""

import logging
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    DATA_DIR,
)
from utils.data_preprocessing import load_and_preprocess
from utils.ui_theme import apply_enterprise_theme
from utils.validation import validate_schema, raise_if_invalid
from models.model_manager import get_model_manager


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("supplyguard")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_enterprise_theme()


# ---------------------------------------------------------------------------
# Cached functions
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_preprocess_cached(source: Any) -> Tuple[pd.DataFrame, dict, Any]:
    """Cache the (expensive) preprocessing step."""
    return load_and_preprocess(source)


@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Cache CSV serialization for the download button."""
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Navigation map — matches the actual files in pages/
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Home": "app.py",
    "📊 Executive Dashboard": "pages/1_Executive_Dashboard.py",
    "📈 Demand Forecasting": "pages/2_Demand_Forecasting.py",
    "⚠️ Stockout Risk": "pages/3_Stockout_Risk.py",
    "🏭 Supplier Risk": "pages/4_Supplier_Risk.py",
    "🔍 Anomaly Detection": "pages/5_Anomaly_Detection.py",
    "📦 Warehouse Optimization": "pages/6_Warehouse_Optimization.py",
    "🔄 What-If Simulation": "pages/7_What_If_Simulation.py",
    "🧠 Explainable AI": "pages/8_Explainable_AI.py",
    "🌱 Sustainability": "pages/9_Sustainability.py",
    "📋 Model Evaluation": "pages/10_Model_Evaluation.py",
    "🧪 Research Experiments": "pages/11_Research_Experiments.py",
    "ℹ️ About": "pages/12_About.py",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize session state variables exactly once."""
    defaults = {
        "data_loaded": False,
        "processed_df": None,
        "preprocessing_report": None,
        "preprocessor": None,
        "models_trained": False,
        "training_status": {},
        "filters": {},
        "experiment_log": [],
        "schema_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> Tuple[str, Optional[Any]]:
    """Render the sidebar with dataset controls + navigation."""
    with st.sidebar:
        st.caption(f"v{APP_VERSION}")
        st.markdown("---")

        # Dataset selection
        st.markdown("### 📁 Dataset")
        data_source = st.radio(
            "Data Source",
            ["Use Sample Dataset", "Upload CSV"],
            index=0,
            key="data_source_selection",
        )

        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        for label, path in PAGES.items():
            try:
                st.page_link(path, label=label, use_container_width=True)
            except Exception:
                st.caption(f"• {label} (coming soon)")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption(
            "SupplyGuard is an Explainable AI framework for risk-aware "
            "and sustainable supply chain management."
        )

        return data_source, uploaded_file


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    data_source: str, uploaded_file: Optional[Any]
) -> Tuple[bool, Optional[pd.DataFrame], Optional[dict], Optional[Any]]:
    """Load, preprocess, and validate data. Returns (success, df, report, preprocessor)."""
    try:
        if data_source == "Use Sample Dataset":
            sample_path = DATA_DIR / "sample_supply_chain.csv"
            if not sample_path.exists():
                logger.warning("Sample dataset missing at %s", sample_path)
                st.error(
                    "⚠️ Sample dataset not found. "
                    "Please run `python data/generate_sample_data.py` first."
                )
                return False, None, None, None
            df, report, preprocessor = load_and_preprocess_cached(str(sample_path))
        else:
            if uploaded_file is None:
                st.warning("Choose a CSV file before loading the dataset.")
                return False, None, None, None
            df, report, preprocessor = load_and_preprocess_cached(uploaded_file)

        if df is None or df.empty:
            st.error("The selected dataset contains no usable rows.")
            return False, None, None, None

        # Validate schema
        schema_report = validate_schema(df)
        raise_if_invalid(schema_report)
        if schema_report.get("missing_recommended"):
            st.warning(
                f"Missing recommended columns: "
                f"{schema_report['missing_recommended']}. "
                "Some modules may be unavailable."
            )

        # Persist state
        st.session_state.processed_df = df
        st.session_state.preprocessing_report = report
        st.session_state.preprocessor = preprocessor
        st.session_state.schema_report = schema_report
        st.session_state.data_loaded = True

        logger.info("Loaded dataset: %s rows, %s columns", len(df), len(df.columns))
        return True, df, report, preprocessor

    except FileNotFoundError:
        logger.exception("File not found")
        st.error("📁 File not found. Check the path and try again.")
    except pd.errors.EmptyDataError:
        logger.exception("Empty or malformed CSV")
        st.error("📄 The file is empty or malformed.")
    except ValueError as exc:
        logger.exception("Validation failed")
        st.error(f"⚠️ Data validation failed: {exc}")
    except Exception:
        logger.exception("Unexpected error during data load")
        st.error("❌ Something went wrong while loading data. Check logs for details.")

    return False, None, None, None


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> dict:
    """Train all available models and report status."""
    manager = get_model_manager()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str) -> None:
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

    completed = sum(
        v == "complete" or v.startswith("complete (") for v in status.values()
    )
    unavailable = sum(v.startswith("unavailable:") for v in status.values())
    failed = len(status) - completed - unavailable

    if unavailable or failed:
        st.warning(
            f"{completed} of {len(status)} modules completed; "
            f"{unavailable} unavailable and {failed} failed for this dataset. "
            "See each module's status for the required columns."
        )
    else:
        st.success(
            f"All {completed} model modules are ready. "
            "Open a page from the navigation menu to explore the results."
        )

    return results


# ---------------------------------------------------------------------------
# UI fragments
# ---------------------------------------------------------------------------
def render_hero() -> None:
    """Render the landing hero banner."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;">
            <h1 style="margin:0; font-size:2.2rem;">🛡️ {APP_TITLE}</h1>
            <p style="margin:0.5rem 0 0; opacity:0.9; font-size:1.05rem;">
                {APP_SUBTITLE}
            </p>
            <p style="margin-top:1rem; opacity:0.75; font-size:0.85rem;">
                v{APP_VERSION} • Built with Streamlit, scikit-learn &amp; SHAP
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown("---")
    st.caption(
        f"© 2025 SupplyGuard · v{APP_VERSION} · Built with ❤️ using Streamlit"
    )


def render_welcome() -> None:
    """Render the welcome / feature grid."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Welcome to SupplyGuard")
        st.markdown(
            """
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
            """
        )

    with col2:
        st.markdown("#### Quick Start")
        if st.button("Load Dataset", use_container_width=True):
            with st.spinner("Loading dataset..."):
                success, df, _, _ = load_data(
                    st.session_state.get("data_source_selection", "Use Sample Dataset"),
                    None,
                )
                if success:
                    st.success(f"✅ Dataset loaded: {len(df):,} rows")

        if st.button("Load & Train", use_container_width=True, type="primary"):
            if not st.session_state.data_loaded:
                success, df, _, _ = load_data(
                    st.session_state.get("data_source_selection", "Use Sample Dataset"),
                    None,
                )
                if not success:
                    st.stop()
            else:
                df = st.session_state.processed_df
                st.success(f"Using cached dataset: {len(df):,} rows")
            train_models(df)

        if st.session_state.data_loaded and st.button(
            "Train Models", use_container_width=True
        ):
            train_models(st.session_state.processed_df)


def render_dataset_status() -> None:
    """Render the dataset metrics + preview + download."""
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
            st.markdown(
                f"**Date Range:** {report['date_range'][0]} to {report['date_range'][1]}"
            )

    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(100), use_container_width=True)

    if st.session_state.training_status:
        with st.expander("Model training status"):
            status_rows = [
                {"Model": name.replace("_", " ").title(), "Status": status}
                for name, status in st.session_state.training_status.items()
            ]
            st.dataframe(
                pd.DataFrame(status_rows), hide_index=True, use_container_width=True
            )

    st.download_button(
        "📥 Download Processed Dataset",
        df_to_csv_bytes(df),
        "supplyguard_processed.csv",
        "text/csv",
    )


def render_empty_state() -> None:
    """Guide shown before data is loaded."""
    st.markdown("---")
    st.info(
        "👈 Select a dataset source in the sidebar and click **Load & Train** to begin."
    )

    st.markdown("### 📋 Expected Dataset Format")
    st.markdown(
        """
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
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_hero()
    data_source, uploaded_file = render_sidebar()

    if st.session_state.data_loaded:
        render_dataset_status()
    else:
        render_welcome()
        render_empty_state()

    render_footer()


if __name__ == "__main__":
    main()
