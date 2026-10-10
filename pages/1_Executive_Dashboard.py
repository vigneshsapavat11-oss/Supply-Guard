"""SupplyGuard - Executive Dashboard Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_theme import apply_enterprise_theme, render_plotly, style_risk_table

from models.model_manager import get_model_manager
from utils.visualization import (
    plot_demand_trend, plot_inventory_distribution,
    plot_risk_distribution, plot_warehouse_inventory,
    COLOR_PALETTE
)

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")
apply_enterprise_theme()

st.markdown("# 📊 Executive Dashboard")
st.markdown("High-level overview of supply chain health and risk indicators.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first. Go to the main page and click 'Load & Train'.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

# KPI Metrics Row 1
st.markdown("### Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

n_products = df["product_id"].nunique() if "product_id" in df.columns else len(df)
n_warehouses = df["warehouse"].nunique() if "warehouse" in df.columns else 1
n_suppliers = df["supplier_id"].nunique() if "supplier_id" in df.columns else 1

col1.metric("Total Products", f"{n_products:,}")
col2.metric("Total Warehouses", f"{n_warehouses}")
col3.metric("Total Suppliers", f"{n_suppliers}")
col4.metric("Total Records", f"{len(df):,}")

# Stock-out risk metrics
col1, col2, col3, col4 = st.columns(4)

high_risk_count = 0
stockout_pct = 0
anomaly_count = 0
avg_inventory = df["inventory"].mean() if "inventory" in df.columns else 0
avg_demand = df["demand"].mean() if "demand" in df.columns else 0

if manager.stockout_predictor and manager.stockout_predictor.model is not None:
    try:
        risk_df = manager.stockout_predictor.predict_risk(df)
        high_risk_count = (risk_df["risk_level"].isin(["HIGH", "CRITICAL"])).sum()
        stockout_pct = (risk_df["risk_level"].isin(["HIGH", "CRITICAL"])).mean() * 100
    except Exception:
        pass

if manager.anomaly_detector and manager.anomaly_detector.results:
    try:
        combined = manager.anomaly_detector.get_combined_anomalies()
        if "anomaly_label_consensus" in combined.columns:
            anomaly_count = (combined["anomaly_label_consensus"] == -1).sum()
    except Exception:
        pass

col1.metric("High-Risk Products", f"{high_risk_count:,}")
col2.metric("Stock-Out Risk %", f"{stockout_pct:.1f}%")
col3.metric("Anomalies Detected", f"{anomaly_count:,}")
col4.metric("Avg Inventory", f"{avg_inventory:,.0f}")

# Charts
st.markdown("---")
st.markdown("### 📈 Demand & Forecast")

col1, col2 = st.columns(2)

with col1:
    if "date" in df.columns and "demand" in df.columns:
        # Aggregate daily demand
        daily = df.groupby("date")["demand"].sum().reset_index()
        fig = px.line(daily, x="date", y="demand", title="Total Demand Trend")
        fig.update_layout(template="plotly_white", height=350)
        render_plotly(fig, use_container_width=True)

with col2:
    if "date" in df.columns and "demand" in df.columns:
        # Show forecast if available
        if manager.demand_forecaster and manager.demand_forecaster.best_model is not None:
            try:
                results = manager.demand_forecaster.results.get(
                    manager.demand_forecaster.best_model_name, {}
                )
                if "y_test" in results and "y_pred" in results:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=results["y_test"][:100], mode="lines", name="Actual"))
                    fig.add_trace(go.Scatter(y=results["y_pred"][:100], mode="lines", name="Predicted", 
                                            line=dict(dash="dash")))
                    fig.update_layout(title="Actual vs Predicted Demand (Test Set)",
                                     template="plotly_white", height=350)
                    render_plotly(fig, use_container_width=True)
            except Exception:
                st.info("Forecast visualization not available")
        else:
            st.info("Train models to see forecast")

st.markdown("---")
st.markdown("### 📦 Inventory & Risk Distribution")

col1, col2 = st.columns(2)

with col1:
    if "warehouse" in df.columns and "inventory" in df.columns:
        fig = plot_warehouse_inventory(df)
        fig.update_layout(height=350)
        render_plotly(fig, use_container_width=True)

with col2:
    if manager.stockout_predictor and manager.stockout_predictor.model is not None:
        try:
            risk_df = manager.stockout_predictor.predict_risk(df)
            fig = plot_risk_distribution(risk_df, "risk_level", "Stock-Out Risk Distribution")
            fig.update_layout(height=350)
            render_plotly(fig, use_container_width=True)
        except Exception:
            st.info("Stock-out risk not available")

st.markdown("---")
st.markdown("### 🏭 Supplier Risk Overview")

if manager.supplier_predictor and manager.supplier_predictor.model is not None:
    try:
        supplier_risk = manager.supplier_predictor.predict_supplier_risk()
        supplier_table = supplier_risk[["supplier_id", "risk_probability", "risk_category", "mitigation"]]
        st.dataframe(style_risk_table(supplier_table, "risk_category"), use_container_width=True)
    except Exception as e:
        st.info(f"Supplier risk not available: {e}")
else:
    st.info("Train models to see supplier risk analysis")