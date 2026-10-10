"""SupplyGuard - What-If Simulation Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager

st.set_page_config(page_title="What-If Simulation", page_icon="🔄", layout="wide")
apply_enterprise_theme()

st.markdown("# 🔄 What-If Simulation")
st.markdown("Explore how changes in supply chain parameters affect risk predictions.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

if manager.stockout_predictor is None or manager.stockout_predictor.model is None:
    st.warning("⚠️ Models not trained. Please train models from the main page.")
    st.stop()

# Select product/warehouse
st.markdown("### 🎯 Select Scenario Base")
col1, col2 = st.columns(2)

with col1:
    if "product_id" in df.columns:
        product = st.selectbox("Product", sorted(df["product_id"].unique()))
    else:
        product = None

with col2:
    if "warehouse" in df.columns:
        warehouse = st.selectbox("Warehouse", sorted(df["warehouse"].unique()))
    else:
        warehouse = None

# Filter base data
baseline_df = df.copy()
if product and "product_id" in baseline_df.columns:
    baseline_df = baseline_df[baseline_df["product_id"] == product]
if warehouse and "warehouse" in baseline_df.columns:
    baseline_df = baseline_df[baseline_df["warehouse"] == warehouse]

if len(baseline_df) == 0:
    st.warning("No data for selected product/warehouse combination.")
    st.stop()

# Baseline values
baseline_row = baseline_df.iloc[-1].to_dict()

st.markdown("### ⚙️ Scenario Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    demand_change = st.slider("Demand Change (%)", -50, 100, 0, 5)
    price_change = st.slider("Price Change (%)", -30, 50, 0, 5)

with col2:
    lead_time_change = st.slider("Lead Time Change (days)", -5, 15, 0, 1)
    inventory_change = st.slider("Inventory Change (%)", -50, 100, 0, 5)

with col3:
    promotion = st.selectbox("Promotion", [0, 1], format_func=lambda x: "ON" if x else "OFF")
    competitor_change = st.slider("Competitor Price Change (%)", -30, 50, 0, 5)

# Create scenario
scenario_df = baseline_df.copy()

if "demand" in scenario_df.columns:
    scenario_df["demand"] = scenario_df["demand"] * (1 + demand_change / 100)
if "price" in scenario_df.columns:
    scenario_df["price"] = scenario_df["price"] * (1 + price_change / 100)
if "lead_time" in scenario_df.columns:
    scenario_df["lead_time"] = scenario_df["lead_time"] + lead_time_change
if "inventory" in scenario_df.columns:
    scenario_df["inventory"] = scenario_df["inventory"] * (1 + inventory_change / 100)
if "promotion" in scenario_df.columns:
    scenario_df["promotion"] = promotion
if "competitor_price" in scenario_df.columns:
    scenario_df["competitor_price"] = scenario_df["competitor_price"] * (1 + competitor_change / 100)

# Update lag features for scenario (simplified)
if "lag_1" in scenario_df.columns and "demand" in scenario_df.columns:
    scenario_df["lag_1"] = scenario_df["demand"].shift(1).fillna(scenario_df["demand"].mean())
if "rolling_mean_7" in scenario_df.columns and "demand" in scenario_df.columns:
    scenario_df["rolling_mean_7"] = scenario_df["demand"].rolling(7, min_periods=1).mean()

# Predictions
st.markdown("### 📊 Baseline vs Scenario Comparison")

try:
    baseline_pred = manager.stockout_predictor.predict_risk(baseline_df)
    scenario_pred = manager.stockout_predictor.predict_risk(scenario_df)
    
    baseline_risk = baseline_pred["stockout_probability"].mean()
    scenario_risk = scenario_pred["stockout_probability"].mean()
    
    baseline_demand = baseline_df["demand"].mean() if "demand" in baseline_df.columns else 0
    scenario_demand = scenario_df["demand"].mean() if "demand" in scenario_df.columns else 0
    
    # Comparison table
    comparison_data = {
        "Metric": ["Average Demand", "Stock-Out Probability", "Risk Level"],
        "Baseline": [
            f"{baseline_demand:,.1f}",
            f"{baseline_risk:.2%}",
            baseline_pred["risk_level"].mode().iloc[0] if len(baseline_pred) > 0 else "N/A"
        ],
        "Scenario": [
            f"{scenario_demand:,.1f}",
            f"{scenario_risk:.2%}",
            scenario_pred["risk_level"].mode().iloc[0] if len(scenario_pred) > 0 else "N/A"
        ],
        "Change": [
            f"{(scenario_demand - baseline_demand) / max(baseline_demand, 1) * 100:+.1f}%",
            f"{(scenario_risk - baseline_risk) * 100:+.2f} pp",
            "↑" if scenario_risk > baseline_risk else "↓" if scenario_risk < baseline_risk else "="
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Gauge comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=baseline_risk * 100,
            title={"text": "Baseline Risk"},
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#3498db"}}
        ))
        fig.update_layout(height=250, template="plotly_white")
        render_plotly(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scenario_risk * 100,
            title={"text": "Scenario Risk"},
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#e74c3c" if scenario_risk > baseline_risk else "#2ecc71"}}
        ))
        fig.update_layout(height=250, template="plotly_white")
        render_plotly(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("### 📝 Interpretation")
    if scenario_risk > baseline_risk:
        st.warning(f"⚠️ The scenario increases stock-out risk by {(scenario_risk - baseline_risk) * 100:.2f} percentage points.")
    elif scenario_risk < baseline_risk:
        st.success(f"✅ The scenario reduces stock-out risk by {(baseline_risk - scenario_risk) * 100:.2f} percentage points.")
    else:
        st.info("The scenario does not change stock-out risk significantly.")
    
    st.caption("*Note: This is a simulation based on model predictions. Actual outcomes may differ.*")
    
except Exception as e:
    st.error(f"Simulation failed: {e}")