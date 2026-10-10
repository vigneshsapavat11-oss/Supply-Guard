"""SupplyGuard - Sustainability Page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager
from utils.sustainability import (
    compute_sustainability_indicators,
    get_sustainability_recommendations
)
from utils.visualization import plot_sustainability_indicators

st.set_page_config(page_title="Sustainability", page_icon="🌱", layout="wide")
apply_enterprise_theme()

st.markdown("# 🌱 Sustainability Analysis")
st.markdown("Relative sustainability indicators for supply chain operations.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

# Get transfer recommendations if available
transfer_df = None
if manager.warehouse_optimizer and manager.warehouse_optimizer.recommendations is not None:
    transfer_df = manager.warehouse_optimizer.recommendations

# Compute indicators
indicators = compute_sustainability_indicators(df, transfer_df)

st.markdown("### 📊 Sustainability Indicators")
st.caption("⚠️ These are RELATIVE indicators, not scientifically validated environmental measurements.")

col1, col2 = st.columns([1, 1])

with col1:
    for name, value in indicators.items():
        if name != "Composite Sustainability Score":
            st.metric(name, f"{value:.4f}")
    
    st.markdown("---")
    st.metric("Composite Sustainability Score", 
              f"{indicators.get('Composite Sustainability Score', 0):.4f}")

with col2:
    # Radar chart
    radar_indicators = {k: v for k, v in indicators.items() 
                       if k != "Composite Sustainability Score"}
    if radar_indicators:
        fig = plot_sustainability_indicators(radar_indicators)
        render_plotly(fig, use_container_width=True)

# Recommendations
st.markdown("### 💡 Recommendations")
recommendations = get_sustainability_recommendations(indicators)
for rec in recommendations:
    st.markdown(rec)

# Distribution of inventory waste risk
st.markdown("### 📊 Inventory Waste Risk Distribution")
if "inventory" in df.columns and "demand" in df.columns:
    df_plot = df.copy()
    df_plot["waste_risk"] = df_plot["inventory"] / (df_plot["demand"] + 1)
    df_plot["waste_category"] = pd.cut(
        df_plot["waste_risk"],
        bins=[0, 1, 2, 3, 5, float("inf")],
        labels=["Very Low", "Low", "Moderate", "High", "Very High"]
    )
    
    waste_counts = df_plot["waste_category"].value_counts().reset_index()
    waste_counts.columns = ["Waste Risk", "Count"]
    
    import plotly.express as px
    fig = px.bar(waste_counts, x="Waste Risk", y="Count",
                 title="Inventory Waste Risk Distribution",
                 color="Waste Risk")
    fig.update_layout(template="plotly_white")
    render_plotly(fig, use_container_width=True)

# Methodology note
st.markdown("---")
st.markdown("### 📝 Methodology")
st.markdown("""
**Sustainability Indicators (Relative):**

1. **Inventory Waste Risk**: Ratio of inventory to demand. Higher values indicate excess stock that may become waste.

2. **Transportation Burden**: Based on total recommended transfer quantity. Normalized against a reference value.

3. **Emergency Replenishment Burden**: Proportion of products with critically low inventory (< 3 days).

4. **Redistribution Efficiency**: Estimated improvement from recommended transfers.

**Composite Score**: Weighted average of all indicators (weights configurable in `config.py`).

*These indicators are RELATIVE and for decision-support purposes only. They do not represent actual carbon emissions or environmental impact measurements.*
""")