"""SupplyGuard - Model Evaluation Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
apply_enterprise_theme()

st.markdown("# 📊 Model Evaluation")
st.markdown("Comprehensive evaluation of all trained models.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

# Tabs for different models
tab1, tab2, tab3 = st.tabs(["📈 Demand Forecasting", "⚠️ Stock-Out", "🏭 Supplier Risk"])

with tab1:
    st.markdown("### Demand Forecasting Evaluation")
    if manager.demand_forecaster and manager.demand_forecaster.results:
        forecaster = manager.demand_forecaster
        
        # Comparison table
        comparison = forecaster.get_comparison_table()
        if not comparison.empty:
            st.dataframe(comparison, use_container_width=True)
        
        # Best model metrics
        results = forecaster.results.get(forecaster.best_model_name, {})
        if "test_metrics" in results:
            st.markdown(f"#### Best Model: {forecaster.best_model_name}")
            metrics = results["test_metrics"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
            col2.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
            col3.metric("R²", f"{metrics.get('R2', 0):.4f}")
            col4.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("WAPE", f"{metrics.get('WAPE', 0):.2f}%")
            metric_col2.metric("sMAPE", f"{metrics.get('sMAPE', 0):.2f}%")
            metric_col3.metric("MASE", f"{metrics.get('MASE', 0):.4f}")
        
        # Residual plot
        if "y_test" in results:
            residuals = results["y_test"] - results["y_pred"]
            fig = px.scatter(x=results["y_pred"], y=residuals, 
                           title="Residual Plot",
                           labels={"x": "Predicted", "y": "Residual"})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(template="plotly_white")
            render_plotly(fig, use_container_width=True)
    else:
        st.info("Demand forecasting model not trained.")

with tab2:
    st.markdown("### Stock-Out Model Evaluation")
    if manager.stockout_predictor and manager.stockout_predictor.results:
        predictor = manager.stockout_predictor
        
        comparison = predictor.get_comparison_table()
        if not comparison.empty:
            st.dataframe(comparison, use_container_width=True)
        
        # Best model metrics
        if predictor.metrics:
            st.markdown(f"#### Selected Model: {predictor.model_name}")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{predictor.metrics.get('Accuracy', 0):.4f}")
            col2.metric("Precision", f"{predictor.metrics.get('Precision', 0):.4f}")
            col3.metric("Recall", f"{predictor.metrics.get('Recall', 0):.4f}")
            col4.metric("F1", f"{predictor.metrics.get('F1', 0):.4f}")
            col5.metric("ROC-AUC", f"{predictor.metrics.get('ROC_AUC', 0):.4f}")
            
            # Confusion matrix
            cm = predictor.metrics.get("Confusion_Matrix")
            if cm:
                cm_array = np.array(cm)
                fig = px.imshow(cm_array, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              title="Confusion Matrix",
                              color_continuous_scale="Blues")
                render_plotly(fig, use_container_width=True)
            
            # ROC curve
            results = predictor.results.get(predictor.model_name, {})
            if "y_test" in results and "y_prob" in results:
                fpr, tpr, _ = roc_curve(results["y_test"], results["y_prob"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", 
                                        name="Random", line=dict(dash="dash")))
                fig.update_layout(title="ROC Curve", template="plotly_white",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate")
                render_plotly(fig, use_container_width=True)
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(results["y_test"], results["y_prob"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
                fig.update_layout(title="Precision-Recall Curve", template="plotly_white",
                                xaxis_title="Recall", yaxis_title="Precision")
                render_plotly(fig, use_container_width=True)
    else:
        st.info("Stock-out model not trained.")

with tab3:
    st.markdown("### Supplier Risk Model Evaluation")
    if manager.supplier_predictor and manager.supplier_predictor.metrics:
        predictor = manager.supplier_predictor
        
        if predictor.metrics:
            st.markdown(f"#### Selected Model: {predictor.model_name}")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{predictor.metrics.get('Accuracy', 0):.4f}")
            col2.metric("Precision", f"{predictor.metrics.get('Precision', 0):.4f}")
            col3.metric("Recall", f"{predictor.metrics.get('Recall', 0):.4f}")
            col4.metric("F1", f"{predictor.metrics.get('F1', 0):.4f}")
            col5.metric("ROC-AUC", f"{predictor.metrics.get('ROC_AUC', 0):.4f}")
            
            cm = predictor.metrics.get("Confusion_Matrix")
            if cm:
                cm_array = np.array(cm)
                fig = px.imshow(cm_array, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              title="Confusion Matrix",
                              color_continuous_scale="Blues")
                render_plotly(fig, use_container_width=True)
    else:
        st.info("Supplier risk model not trained.")