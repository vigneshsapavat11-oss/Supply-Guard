"""
SupplyGuard - Visualization Module
Plotly-based charts for supply chain analytics.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List


COLOR_PALETTE = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH": "#e74c3c",
    "CRITICAL": "#8e44ad",
    "primary": "#3498db",
    "secondary": "#2c3e50",
    "accent": "#1abc9c"
}


def plot_demand_trend(df: pd.DataFrame, date_col: str = "date",
                      demand_col: str = "demand",
                      product: str = None, warehouse: str = None) -> go.Figure:
    """Plot demand trend over time."""
    plot_df = df.copy()
    if product and "product_id" in plot_df.columns:
        plot_df = plot_df[plot_df["product_id"] == product]
    if warehouse and "warehouse" in plot_df.columns:
        plot_df = plot_df[plot_df["warehouse"] == warehouse]
    
    fig = px.line(plot_df, x=date_col, y=demand_col,
                  title="Demand Trend Over Time",
                  labels={date_col: "Date", demand_col: "Demand"})
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


def plot_actual_vs_predicted(dates, actual, predicted,
                              title: str = "Actual vs Predicted Demand") -> go.Figure:
    """Plot actual vs predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines",
                             name="Actual", line=dict(color=COLOR_PALETTE["primary"])))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines",
                             name="Predicted", line=dict(color=COLOR_PALETTE["HIGH"], dash="dash")))
    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="Date", yaxis_title="Demand", hovermode="x unified")
    return fig


def plot_risk_gauge(probability: float, title: str = "Risk Probability") -> go.Figure:
    """Create a gauge chart for risk probability."""
    if probability < 0.25:
        color = COLOR_PALETTE["LOW"]
    elif probability < 0.50:
        color = COLOR_PALETTE["MEDIUM"]
    elif probability < 0.75:
        color = COLOR_PALETTE["HIGH"]
    else:
        color = COLOR_PALETTE["CRITICAL"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": title},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#d5f5e3"},
                {"range": [25, 50], "color": "#fdebd0"},
                {"range": [50, 75], "color": "#fadbd8"},
                {"range": [75, 100], "color": "#e8daef"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(template="plotly_white", height=300)
    return fig


def plot_risk_distribution(df: pd.DataFrame, risk_col: str,
                           title: str = "Risk Distribution") -> go.Figure:
    """Plot distribution of risk levels."""
    counts = df[risk_col].value_counts().reset_index()
    counts.columns = ["Risk Level", "Count"]
    
    colors = [COLOR_PALETTE.get(r, "#95a5a6") for r in counts["Risk Level"]]
    
    fig = px.bar(counts, x="Risk Level", y="Count", title=title,
                 color="Risk Level", color_discrete_map=COLOR_PALETTE)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_anomalies(df: pd.DataFrame, date_col: str = "date",
                   demand_col: str = "demand",
                   anomaly_col: str = "anomaly_label") -> go.Figure:
    """Plot demand with anomalies highlighted."""
    fig = go.Figure()
    
    normal = df[df[anomaly_col] == 1]
    anomalies = df[df[anomaly_col] == -1]
    
    fig.add_trace(go.Scatter(x=normal[date_col], y=normal[demand_col],
                             mode="markers", name="Normal",
                             marker=dict(color=COLOR_PALETTE["primary"], size=4)))
    fig.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies[demand_col],
                             mode="markers", name="Anomaly",
                             marker=dict(color=COLOR_PALETTE["HIGH"], size=10, symbol="x")))
    
    fig.update_layout(title="Demand Anomaly Detection", template="plotly_white",
                      xaxis_title="Date", yaxis_title="Demand", hovermode="x unified")
    return fig


def plot_inventory_distribution(df: pd.DataFrame, inventory_col: str = "inventory",
                                 warehouse_col: str = "warehouse") -> go.Figure:
    """Plot inventory distribution across warehouses."""
    if warehouse_col in df.columns:
        fig = px.box(df, x=warehouse_col, y=inventory_col,
                     title="Inventory Distribution by Warehouse",
                     color=warehouse_col)
    else:
        fig = px.histogram(df, x=inventory_col, title="Inventory Distribution")
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_supplier_risk(df: pd.DataFrame, supplier_col: str = "supplier_id",
                       risk_col: str = "risk_probability") -> go.Figure:
    """Plot supplier risk comparison."""
    fig = px.bar(df, x=supplier_col, y=risk_col,
                 title="Supplier Risk Comparison",
                 color=risk_col, color_continuous_scale="RdYlGn_r")
    fig.update_layout(template="plotly_white")
    return fig


def plot_warehouse_inventory(df: pd.DataFrame, warehouse_col: str = "warehouse",
                              inventory_col: str = "inventory") -> go.Figure:
    """Plot warehouse inventory levels."""
    agg = df.groupby(warehouse_col)[inventory_col].sum().reset_index()
    fig = px.bar(agg, x=warehouse_col, y=inventory_col,
                 title="Total Inventory by Warehouse",
                 color=inventory_col, color_continuous_scale="Blues")
    fig.update_layout(template="plotly_white")
    return fig


def plot_shap_summary(shap_values, features, feature_names: List[str]) -> go.Figure:
    """Create a SHAP-like summary plot using Plotly."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=True)
    
    fig = px.bar(importance_df, x="Mean |SHAP|", y="Feature", orientation="h",
                 title="Global Feature Importance (Mean |SHAP|)")
    fig.update_layout(template="plotly_white", height=max(400, len(feature_names) * 25))
    return fig


def plot_sustainability_indicators(indicators: dict) -> go.Figure:
    """Plot sustainability indicators as a radar chart."""
    categories = list(indicators.keys())
    values = list(indicators.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Sustainability Indicators"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Relative Sustainability Indicators",
        template="plotly_white"
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                            title: str = "Feature Importance") -> go.Figure:
    """Plot feature importance."""
    fig = px.bar(importance_df.head(20), x="Importance", y="Feature",
                 orientation="h", title=title)
    fig.update_layout(template="plotly_white",
                      height=max(400, min(20, len(importance_df)) * 25))
    return fig