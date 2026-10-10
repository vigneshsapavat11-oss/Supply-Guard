"""
SupplyGuard - SHAP Explainability Module
Provides global and local explanations for ML predictions.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go


def check_shap_available() -> bool:
    """Check if SHAP is available."""
    try:
        import shap
        return True
    except ImportError:
        return False


def _select_output_values(shap_values: np.ndarray) -> np.ndarray:
    """Select the positive class for SHAP arrays with an output dimension."""
    values = np.asarray(shap_values)
    if values.ndim == 3:
        output_index = 1 if values.shape[-1] > 1 else 0
        values = values[:, :, output_index]
    return values


def compute_shap_values(model, X, feature_names: List[str],
                        model_type: str = "tree") -> Optional[np.ndarray]:
    """
    Compute SHAP values for a model.
    Returns None if SHAP is not available or fails.
    """
    if not check_shap_available():
        st.warning("⚠️ SHAP library is not available. Install with: pip install shap")
        return None
    
    try:
        import shap
        
        X_frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)
        X_frame = X_frame.reindex(columns=feature_names)

        pipeline = model if hasattr(model, "named_steps") else None
        actual_model = pipeline.named_steps.get("model", model) if pipeline is not None else model
        X_model = pipeline[:-1].transform(X_frame) if pipeline is not None else X_frame.to_numpy()
        background_count = min(50, len(X_model))
        background_indices = np.linspace(0, len(X_model) - 1, background_count, dtype=int)
        background = X_model[background_indices]

        if model_type == "tree":
            explainer = shap.TreeExplainer(
                actual_model,
                data=background,
                feature_perturbation="interventional",
            )
            shap_values = explainer.shap_values(X_model)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            return _select_output_values(shap_values)

        def predict_output(values):
            frame = pd.DataFrame(values, columns=feature_names)
            if pipeline is not None:
                return pipeline.predict_proba(frame)[:, 1]
            if hasattr(model, "predict_proba"):
                return model.predict_proba(frame)[:, 1]
            return model.predict(frame)

        background_frame = X_frame.iloc[background_indices]
        explainer = shap.Explainer(
            predict_output,
            shap.maskers.Independent(background_frame),
        )
        explanation = explainer(X_frame)
        return _select_output_values(explanation.values)
    except Exception as e:
        st.warning(f"⚠️ SHAP computation failed: {e}")
        return None


def get_global_feature_importance(shap_values: np.ndarray,
                                   feature_names: List[str]) -> pd.DataFrame:
    """Get global feature importance from SHAP values."""
    if shap_values is None:
        return pd.DataFrame()
    
    values = np.asarray(shap_values)
    if values.ndim == 0:
        return pd.DataFrame()

    feature_axis = 1 if values.ndim > 1 else 0
    if values.shape[feature_axis] != len(feature_names):
        matching_axes = [axis for axis, size in enumerate(values.shape)
                         if size == len(feature_names)]
        if len(matching_axes) != 1:
            raise ValueError(
                "SHAP values do not contain a uniquely identifiable feature axis "
                f"for {len(feature_names)} feature names (shape: {values.shape})."
            )
        feature_axis = matching_axes[0]

    reduction_axes = tuple(axis for axis in range(values.ndim) if axis != feature_axis)
    mean_abs = np.abs(values).mean(axis=reduction_axes) if reduction_axes else np.abs(values)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_abs
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    return importance_df


def explain_local_prediction(shap_values: np.ndarray, X_instance,
                             feature_names: List[str],
                             instance_idx: int = 0) -> pd.DataFrame:
    """
    Get local explanation for a single prediction.
    Returns a DataFrame with feature contributions.
    """
    if shap_values is None:
        return pd.DataFrame()
    
    values = _select_output_values(shap_values)
    if instance_idx >= len(values):
        instance_idx = 0
    
    contributions = values[instance_idx]
    
    if isinstance(X_instance, pd.DataFrame):
        actual_values = X_instance.iloc[instance_idx].values
    else:
        actual_values = X_instance[instance_idx]
    
    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Actual_Value": actual_values,
        "SHAP_Contribution": contributions,
        "Direction": ["Increases Risk" if c > 0 else "Decreases Risk" for c in contributions],
        "Abs_Contribution": np.abs(contributions)
    }).sort_values("Abs_Contribution", ascending=False).reset_index(drop=True)
    
    return explanation_df


def generate_natural_language_explanation(explanation_df: pd.DataFrame,
                                          risk_level: str,
                                          top_n: int = 5) -> str:
    """
    Generate natural-language explanation from SHAP values.
    Uses careful wording: "contributed to" not "caused".
    """
    if explanation_df.empty:
        return "Explanation not available."
    
    top_features = explanation_df.head(top_n)
    
    increasing = top_features[top_features["SHAP_Contribution"] > 0]
    decreasing = top_features[top_features["SHAP_Contribution"] < 0]
    
    text = f"**Prediction: {risk_level} Risk**\n\n"
    text += "**Features that contributed to higher risk:**\n"
    
    if len(increasing) > 0:
        for _, row in increasing.iterrows():
            text += f"- **{row['Feature']}** = {row['Actual_Value']:.2f} "
            text += f"(contribution: +{row['SHAP_Contribution']:.4f})\n"
    else:
        text += "- No features pushed risk upward significantly.\n"
    
    text += "\n**Features that contributed to lower risk:**\n"
    
    if len(decreasing) > 0:
        for _, row in decreasing.iterrows():
            text += f"- **{row['Feature']}** = {row['Actual_Value']:.2f} "
            text += f"(contribution: {row['SHAP_Contribution']:.4f})\n"
    else:
        text += "- No features pushed risk downward significantly.\n"
    
    text += "\n*Note: These features contributed to the model's prediction. "
    text += "This does not imply causality.*"
    
    return text


def answer_explainability_chat(question: str, explanation_df: pd.DataFrame,
                               probability: Optional[float] = None,
                               global_importance: Optional[pd.DataFrame] = None) -> str:
    """Answer explainability questions or retrieve relevant project sources."""
    normalized_question = question.lower()
    positive = explanation_df[explanation_df["SHAP_Contribution"] > 0].head(3)
    negative = explanation_df[explanation_df["SHAP_Contribution"] < 0].head(3)

    def describe(rows: pd.DataFrame) -> str:
        if rows.empty:
            return "no features in this explanation"
        return ", ".join(
            f"{row['Feature']} ({row['SHAP_Contribution']:+.3f})"
            for _, row in rows.iterrows()
        )

    if any(term in normalized_question for term in ("cause", "caused", "causal")):
        return "No. SHAP shows how the model used each feature; it does not show that a feature caused the risk."

    if "global" in normalized_question or "overall" in normalized_question:
        if global_importance is not None and not global_importance.empty:
            top_features = global_importance.head(5)
            summary = ", ".join(
                f"{row['Feature']} ({row['Importance']:.3f})"
                for _, row in top_features.iterrows()
            )
            return (
                f"The strongest global contributors by mean absolute SHAP value are {summary}. "
                "Global importance summarizes magnitude across samples, not contribution direction for this one record."
            )
        return "Global importance summarizes average absolute SHAP contributions across the analyzed samples."

    if any(term in normalized_question for term in ("probability", "confidence", "score")):
        if probability is None:
            return "A probability score is not available for this sample."
        return (
            f"The model reports {probability:.1%} for this sample. This is a model estimate, not a guarantee; "
            "it should not be treated as calibrated without a separate calibration evaluation."
        )

    if any(term in normalized_question for term in ("increase", "higher", "raise", "upward", "positive")):
        return f"The strongest upward contributions are {describe(positive)}."

    if any(term in normalized_question for term in ("decrease", "lower", "reduce", "downward", "negative")):
        return f"The strongest downward contributions are {describe(negative)}."

    if any(term in normalized_question for term in ("shap", "contribution", "mean", "value", "sign")):
        return (
            "A SHAP contribution describes how a feature shifts the model output relative to its baseline. "
            "The sign shows direction; the numeric units depend on the model and explainer and are not automatically percentage points."
        )

    project_terms = (
        "project", "module", "code", "architecture", "dataset", "preprocess", "forecast",
        "stockout", "stock-out", "supplier", "warehouse", "anomaly", "sustainab", "experiment",
        "application", "app", "algorithm", "technology", "train", "training",
    )
    if any(term in normalized_question for term in project_terms):
        from utils.project_knowledge import search_project_knowledge

        project_answer = search_project_knowledge(question)
        if project_answer:
            return project_answer

    if any(term in normalized_question for term in ("change", "alter", "modify", "affect the prediction")):
        return "No. This chat only interprets the displayed results; it does not change the model, inputs, prediction, or SHAP values."

    if any(term in normalized_question for term in ("why", "explain", "prediction", "risk")):
        return (
            f"For this sample, upward contributions include {describe(positive)}; "
            f"downward contributions include {describe(negative)}. These explain the model output, not real-world causation."
        )

    from utils.project_knowledge import search_project_knowledge

    project_answer = search_project_knowledge(question)
    if project_answer:
        return project_answer
    return "I couldn't find a relevant answer in the project files. Try asking about a project module, model, dataset, or method."


def plot_local_explanation(explanation_df: pd.DataFrame,
                           top_n: int = 10) -> 'go.Figure':
    """Create a waterfall-style plot for local explanation."""
    import plotly.graph_objects as go
    
    top = explanation_df.head(top_n).sort_values("SHAP_Contribution")
    
    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in top["SHAP_Contribution"]]
    
    fig = go.Figure(go.Bar(
        x=top["SHAP_Contribution"],
        y=top["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.4f}" for c in top["SHAP_Contribution"]],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Local Feature Contributions (SHAP)",
        xaxis_title="SHAP Contribution",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, len(top) * 30)
    )
    
    return fig