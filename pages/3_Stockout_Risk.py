"""SupplyGuard - Stock-Out Risk Page"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.ui_theme import apply_enterprise_theme, render_plotly, style_risk_table
from models.model_manager import get_model_manager
from utils.visualization import plot_risk_gauge, plot_risk_distribution
from config import STOCKOUT_RISK_LEVELS

st.set_page_config(page_title="Stock-Out Risk", page_icon="⚠️", layout="wide")
apply_enterprise_theme()

st.markdown("# ⚠️ Stock-Out Risk Analysis")
st.markdown("Predictive early warning system for inventory depletion risk.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

if manager.stockout_predictor is None or manager.stockout_predictor.model is None:
    st.warning("⚠️ Stock-out model not trained. Please train models from the main page.")
    st.stop()

predictor = manager.stockout_predictor

# Model comparison
st.markdown("### 📊 Model Comparison")
comparison = predictor.get_comparison_table()
if not comparison.empty:
    st.dataframe(comparison, use_container_width=True)
    st.caption(f"**Selected model:** {predictor.model_name} (highest F1 score)")

# Get predictions
try:
    risk_df = predictor.predict_risk(df)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Risk distribution
st.markdown("### 📊 Risk Distribution")
col1, col2 = st.columns([1, 2])

with col1:
    risk_counts = risk_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    st.dataframe(style_risk_table(risk_counts, "Risk Level"), use_container_width=True)

with col2:
    fig = plot_risk_distribution(risk_df, "risk_level", "Stock-Out Risk Distribution")
    render_plotly(fig, use_container_width=True)

# Filters
st.markdown("### 🔍 Product-Level Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    products = ["All"] + sorted(risk_df["product_id"].unique().tolist()) if "product_id" in risk_df.columns else ["All"]
    selected_product = st.selectbox("Product", products)

with col2:
    warehouses = ["All"] + sorted(risk_df["warehouse"].unique().tolist()) if "warehouse" in risk_df.columns else ["All"]
    selected_warehouse = st.selectbox("Warehouse", warehouses)

with col3:
    risk_levels = ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    selected_risk = st.selectbox("Risk Level", risk_levels)

filtered = risk_df.copy()
if selected_product != "All" and "product_id" in filtered.columns:
    filtered = filtered[filtered["product_id"] == selected_product]
if selected_warehouse != "All" and "warehouse" in filtered.columns:
    filtered = filtered[filtered["warehouse"] == selected_warehouse]
if selected_risk != "All":
    filtered = filtered[filtered["risk_level"] == selected_risk]

# Display risk table
display_cols = [c for c in ["product_id", "warehouse", "inventory", "stockout_probability",
                            "risk_level", "days_of_inventory", "recommended_action"] 
                if c in filtered.columns]

if len(filtered) > 0:
    st.dataframe(
        style_risk_table(
            filtered[display_cols].sort_values("stockout_probability", ascending=False).head(100),
            "risk_level",
        ),
        use_container_width=True
    )
    
    # Download
    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Risk Analysis", csv, "stockout_risk.csv", "text/csv")
else:
    st.info("No records match the selected filters.")

# Individual risk gauge
st.markdown("### 🎯 Individual Risk Assessment")
if len(filtered) > 0:
    idx = st.number_input("Select row index", 0, len(filtered) - 1, 0)
    row = filtered.iloc[idx]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = plot_risk_gauge(
            row["stockout_probability"],
            f"Stock-Out Risk: {row.get('product_id', 'N/A')} @ {row.get('warehouse', 'N/A')}"
        )
        render_plotly(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Details")
        st.write(f"**Probability:** {row['stockout_probability']:.2%}")
        st.write(f"**Risk Level:** {row['risk_level']}")
        if "days_of_inventory" in row:
            st.write(f"**Days of Inventory:** {row['days_of_inventory']:.1f}")
        if "inventory" in row:
            st.write(f"**Current Inventory:** {row['inventory']:,}")
        st.write(f"**Recommended Action:** {row.get('recommended_action', 'N/A')}")