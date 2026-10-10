"""SupplyGuard - Demand Forecasting Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager
from config import FORECAST_HORIZONS

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")
apply_enterprise_theme()

st.markdown("# 📈 Forecasting")
st.markdown("Multi-model forecasting with regression metrics and chronological validation.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

if manager.demand_forecaster is None:
    st.warning("⚠️ Demand forecasting model not trained. Please train models from the main page.")
    st.stop()

forecaster = manager.demand_forecaster
target_label = forecaster.target_col.replace("_", " ").title()

# Split info
st.markdown("### 📅 Data Split Information")
split = forecaster.split_info
if split:
    col1, col2, col3 = st.columns(3)
    col1.info(f"**Training:** {split['train_size']} rows\n{split['train_period'][0]} to {split['train_period'][1]}")
    col2.info(f"**Validation:** {split['val_size']} rows\n{split['val_period'][0]} to {split['val_period'][1]}")
    col3.info(f"**Testing:** {split['test_size']} rows\n{split['test_period'][0]} to {split['test_period'][1]}")

# Model comparison
st.markdown("### 📊 Model Comparison")
comparison = forecaster.get_comparison_table()
if not comparison.empty:
    st.dataframe(comparison, use_container_width=True)
    st.caption(f"**Selected model:** {forecaster.best_model_name} (lowest validation RMSE; test metrics are held out)")

lightgbm_tuning = forecaster.results.get("LightGBM", {})
if "cv_rmse" in lightgbm_tuning:
    with st.expander("LightGBM tuning details"):
        st.caption(f"Parameters selected with {lightgbm_tuning['cv_folds']} expanding-window folds inside the training period.")
        st.metric("Training CV RMSE", f"{lightgbm_tuning['cv_rmse']:.2f}")
        st.json(lightgbm_tuning["params"])

# Baseline comparison
if forecaster.baseline_metrics:
    st.markdown("### 📉 Baseline Comparison")
    baseline_rows = []
    for name, metrics in forecaster.baseline_metrics.items():
        baseline_rows.append({
            "Model": name,
            "RMSE": metrics.get("RMSE"),
            "MAE": metrics.get("MAE"),
            "R2": metrics.get("R2")
        })
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True)

# Actual vs Predicted
st.markdown("### 📈 Actual vs Predicted")
results = forecaster.results.get(forecaster.best_model_name, {})
if "y_test" in results:
    n_show = st.slider("Number of test points to display", 50, min(1000, len(results["y_test"])), 200)
    actual = np.asarray(results["y_test"][:n_show])
    predicted = np.asarray(results["y_pred"][:n_show])
    plot_dates = np.asarray(results["test_dates"][:n_show]).reshape(-1, 1)
    value_min = min(actual.min(), predicted.min())
    value_max = max(actual.max(), predicted.max())
    padding = (value_max - value_min) * 0.05 or 1
    axis_min, axis_max = value_min - padding, value_max + padding

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min, axis_max], mode="lines",
        name="Perfect prediction", line=dict(color="#87939d", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=predicted, y=actual, customdata=plot_dates, mode="markers",
        name="Held-out observations", marker=dict(color="#187d72", size=7, opacity=0.65),
        hovertemplate="Date: %{customdata[0]}<br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(title=f"Actual vs Predicted {target_label}", template="plotly_white",
                     xaxis_title=f"Predicted {target_label}", yaxis_title=f"Actual {target_label}", height=450)
    fig.update_xaxes(range=[axis_min, axis_max])
    fig.update_yaxes(range=[axis_min, axis_max], scaleanchor="x", scaleratio=1)
    render_plotly(fig, use_container_width=True)
    
    # Metrics
    st.markdown("#### Test Set Metrics")
    metrics = results["test_metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
    col2.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
    col3.metric("R²", f"{metrics.get('R2', 0):.4f}")
    col4.metric("MAPE", f"{metrics.get('MAPE', 0):.1f}%")
    col5.metric("MSE", f"{metrics.get('MSE', 0):.2f}")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("WAPE", f"{metrics.get('WAPE', 0):.1f}%")
    metric_col2.metric("sMAPE", f"{metrics.get('sMAPE', 0):.1f}%")
    metric_col3.metric("MASE", f"{metrics.get('MASE', 0):.3f}")

# Future forecast
st.markdown("### 🔮 Future Forecast")
date_values = pd.DatetimeIndex(pd.to_datetime(df[forecaster.date_col], errors="coerce").dropna().unique()).sort_values()
frequency = pd.infer_freq(date_values) if len(date_values) >= 3 else None
is_monthly_or_slower = bool(frequency and frequency.upper().startswith(("M", "Q", "A", "Y")))
horizon_options = [1, 3, 6, 12] if is_monthly_or_slower else FORECAST_HORIZONS
horizon = st.selectbox("Forecast Horizon (periods)", horizon_options, index=1)

if st.button("Generate Forecast"):
    try:
        with st.spinner("Generating forecast..."):
            forecast_df = forecaster.forecast_future(df, horizon=horizon)
            st.dataframe(forecast_df, use_container_width=True)
            
            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Forecast", csv, "demand_forecast.csv", "text/csv")
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")

# Feature importance
st.markdown("### 🎯 Feature Importance")
try:
    model = forecaster.best_model
    if hasattr(model, "named_steps"):
        model_step = model.named_steps.get("model")
        if hasattr(model_step, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": forecaster.feature_names,
                "Importance": model_step.feature_importances_
            }).sort_values("Importance", ascending=False).head(20)
            
            import plotly.express as px
            fig = px.bar(importance, x="Importance", y="Feature", orientation="h")
            fig.update_layout(template="plotly_white", height=500, 
                             title="Top 20 Feature Importances")
            render_plotly(fig, use_container_width=True)
except Exception as e:
    st.info("Feature importance not available for this model type.")