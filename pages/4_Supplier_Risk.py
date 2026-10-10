"""SupplyGuard - Supplier Risk Page"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_theme import apply_enterprise_theme, render_plotly, style_risk_table
from models.model_manager import get_model_manager
from utils.visualization import plot_supplier_risk

st.set_page_config(page_title="Supplier Risk", page_icon="🏭", layout="wide")
apply_enterprise_theme()

st.markdown("# 🏭 Supplier Risk Analysis")
st.markdown("Predict supplier-related risk and identify mitigation strategies.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

if manager.supplier_predictor is None or manager.supplier_predictor.model is None:
    st.warning("⚠️ Supplier risk model not trained. Please train models from the main page.")
    st.stop()

predictor = manager.supplier_predictor

# Predictions
try:
    supplier_risk = predictor.predict_supplier_risk()
except Exception as e:
    st.error(f"Supplier risk prediction failed: {e}")
    st.stop()

# Risk distribution
st.markdown("### 📊 Supplier Risk Distribution")
col1, col2 = st.columns(2)

with col1:
    risk_counts = supplier_risk["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Count"]
    st.dataframe(risk_counts, use_container_width=True)

with col2:
    fig = px.pie(supplier_risk, names="risk_category", title="Risk Category Distribution",
                 color="risk_category",
                 color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f39c12",
                                    "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"})
    fig.update_layout(template="plotly_white")
    render_plotly(fig, use_container_width=True)

# Supplier risk table
st.markdown("### 📋 Supplier Risk Details")
display_cols = [c for c in ["supplier_id", "risk_probability", "risk_category",
                            "avg_lead_time", "lead_time_variability",
                            "order_frequency", "mitigation"]
                if c in supplier_risk.columns]

st.dataframe(
    style_risk_table(
        supplier_risk[display_cols].sort_values("risk_probability", ascending=False),
        "risk_category",
    ),
    use_container_width=True
)

# Download
csv = supplier_risk[display_cols].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Supplier Risk", csv, "supplier_risk.csv", "text/csv")

# Risk comparison chart
st.markdown("### 📊 Supplier Risk Comparison")
if "supplier_id" in supplier_risk.columns:
    fig = px.bar(
        supplier_risk.sort_values("risk_probability", ascending=False),
        x="supplier_id", y="risk_probability", color="risk_category",
        title="Supplier Risk Probability",
        color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f39c12",
                           "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
    )
    fig.update_layout(template="plotly_white", height=400)
    render_plotly(fig, use_container_width=True)

# Feature importance
st.markdown("### 🎯 Risk Contributing Factors")
importance = predictor.get_feature_importance()
if not importance.empty:
    fig = px.bar(importance.head(15), x="Importance", y="Feature", orientation="h",
                 title="Feature Importance for Supplier Risk")
    fig.update_layout(template="plotly_white", height=450)
    render_plotly(fig, use_container_width=True)
else:
    st.info("Feature importance not available for this model type.")

# Model metrics
st.markdown("### 📈 Model Performance")
if predictor.metrics:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{predictor.metrics.get('Accuracy', 0):.4f}")
    col2.metric("Precision", f"{predictor.metrics.get('Precision', 0):.4f}")
    col3.metric("Recall", f"{predictor.metrics.get('Recall', 0):.4f}")
    col4.metric("F1 Score", f"{predictor.metrics.get('F1', 0):.4f}")

st.caption("⚠️ Note: Supplier risk labels are derived from operational metrics using transparent rules. "
          "These are simulated labels for demonstration and do not represent actual supplier disruption data.")