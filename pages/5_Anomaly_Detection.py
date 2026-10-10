"""SupplyGuard - Anomaly Detection Page"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager
from utils.visualization import plot_anomalies

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")
apply_enterprise_theme()

st.markdown("# 🔍 Demand Anomaly Detection")
st.markdown("Detect unusual demand patterns using multiple methods.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

if manager.anomaly_detector is None:
    st.warning("⚠️ Anomaly detection not run. Please train models from the main page.")
    st.stop()

detector = manager.anomaly_detector

# Method comparison
st.markdown("### 📊 Method Comparison")
comparison = detector.get_comparison_table()
if not comparison.empty:
    st.dataframe(comparison, use_container_width=True)

# Combined anomalies
combined = detector.get_combined_anomalies()

if not combined.empty:
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        products = ["All"] + sorted(combined["product_id"].unique().tolist()) if "product_id" in combined.columns else ["All"]
        selected_product = st.selectbox("Product", products)
    with col2:
        if "anomaly_type_if" in combined.columns:
            anomaly_types = ["All"] + sorted(combined["anomaly_type_if"].unique().tolist())
            selected_type = st.selectbox("Anomaly Type", anomaly_types)
        else:
            selected_type = "All"
    
    filtered = combined.copy()
    if selected_product != "All" and "product_id" in filtered.columns:
        filtered = filtered[filtered["product_id"] == selected_product]
    if selected_type != "All" and "anomaly_type_if" in filtered.columns:
        filtered = filtered[filtered["anomaly_type_if"] == selected_type]
    
    # Plot anomalies
    st.markdown("### 📈 Demand with Anomalies")
    if "date" in filtered.columns and "demand" in filtered.columns:
        label_col = "anomaly_label_consensus" if "anomaly_label_consensus" in filtered.columns else "anomaly_label_if"
        
        if label_col in filtered.columns:
            import plotly.graph_objects as go
            
            normal = filtered[filtered[label_col] == 1]
            anomalies = filtered[filtered[label_col] == -1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=normal["date"], y=normal["demand"],
                mode="markers", name="Normal",
                marker=dict(color="#3498db", size=4)
            ))
            fig.add_trace(go.Scatter(
                x=anomalies["date"], y=anomalies["demand"],
                mode="markers", name="Anomaly",
                marker=dict(color="#e74c3c", size=10, symbol="x")
            ))
            fig.update_layout(
                title="Demand Anomaly Detection",
                template="plotly_white",
                xaxis_title="Date", yaxis_title="Demand",
                height=450, hovermode="x unified"
            )
            render_plotly(fig, use_container_width=True)
    
    # Anomaly table
    st.markdown("### 📋 Detected Anomalies")
    anomaly_cols = [c for c in ["date", "product_id", "warehouse", "demand",
                                "anomaly_type_if", "anomaly_score_if"]
                    if c in filtered.columns]
    
    anomaly_rows = filtered[filtered.get("anomaly_label_consensus", 
                                          filtered.get("anomaly_label_if", 1)) == -1]
    
    if len(anomaly_rows) > 0:
        st.dataframe(anomaly_rows[anomaly_cols].head(100), use_container_width=True)
        
        csv = anomaly_rows[anomaly_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Anomalies", csv, "anomalies.csv", "text/csv")
    else:
        st.info("No anomalies detected with current filters.")
    
    # Summary stats
    st.markdown("### 📊 Anomaly Summary")
    col1, col2, col3 = st.columns(3)
    
    total_anomalies = len(anomaly_rows)
    col1.metric("Total Anomalies", f"{total_anomalies:,}")
    
    if "anomaly_type_if" in anomaly_rows.columns:
        spikes = (anomaly_rows["anomaly_type_if"] == "Spike").sum()
        drops = (anomaly_rows["anomaly_type_if"] == "Drop").sum()
        col2.metric("Demand Spikes", f"{spikes:,}")
        col3.metric("Demand Drops", f"{drops:,}")