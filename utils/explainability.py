"""
SupplyGuard - SHAP Explainability Module
Provides global and local explanations for ML predictions,
plus a grounded, project-aware explainability chat with all 15 questions visible.
"""

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    import plotly.graph_objects as go


# =============================================================================
# Availability check
# =============================================================================
def check_shap_available() -> bool:
    """Check if SHAP is available."""
    try:
        import shap  # noqa: F401
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


# =============================================================================
# SHAP computation
# =============================================================================
def compute_shap_values(
    model,
    X,
    feature_names: List[str],
    model_type: str = "tree",
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for a model.
    Returns None if SHAP is unavailable or computation fails.
    """
    if not check_shap_available():
        st.warning("⚠️ SHAP library is not available. Install with: pip install shap")
        return None

    try:
        import shap

        X_frame = (
            X.copy()
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=feature_names)
        )
        X_frame = X_frame.reindex(columns=feature_names)

        pipeline = model if hasattr(model, "named_steps") else None
        actual_model = (
            pipeline.named_steps.get("model", model)
            if pipeline is not None
            else model
        )
        X_model = (
            pipeline[:-1].transform(X_frame)
            if pipeline is not None
            else X_frame.to_numpy()
        )

        background_count = min(50, len(X_model))
        background_indices = np.linspace(
            0, len(X_model) - 1, background_count, dtype=int
        )
        background = X_model[background_indices]

        if model_type == "tree":
            explainer = shap.TreeExplainer(
                actual_model,
                data=background,
                feature_perturbation="interventional",
            )
            shap_values = explainer.shap_values(
                X_model,
                check_additivity=False,
            )
            if isinstance(shap_values, list):
                shap_values = (
                    shap_values[1] if len(shap_values) > 1 else shap_values[0]
                )
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


# =============================================================================
# Global importance
# =============================================================================
def get_global_feature_importance(
    shap_values: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    """Get global feature importance from SHAP values."""
    if shap_values is None:
        return pd.DataFrame()

    values = np.asarray(shap_values)
    if values.ndim == 0:
        return pd.DataFrame()

    feature_axis = 1 if values.ndim > 1 else 0
    if values.shape[feature_axis] != len(feature_names):
        matching_axes = [
            axis
            for axis, size in enumerate(values.shape)
            if size == len(feature_names)
        ]
        if len(matching_axes) != 1:
            raise ValueError(
                "SHAP values do not contain a uniquely identifiable feature axis "
                f"for {len(feature_names)} feature names (shape: {values.shape})."
            )
        feature_axis = matching_axes[0]

    reduction_axes = tuple(
        axis for axis in range(values.ndim) if axis != feature_axis
    )
    mean_abs = (
        np.abs(values).mean(axis=reduction_axes)
        if reduction_axes
        else np.abs(values)
    )
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": mean_abs})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


# =============================================================================
# Local explanation
# =============================================================================
def explain_local_prediction(
    shap_values: np.ndarray,
    X_instance,
    feature_names: List[str],
    instance_idx: int = 0,
) -> pd.DataFrame:
    """Get local explanation for a single prediction."""
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

    explanation_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Actual_Value": actual_values,
                "SHAP_Contribution": contributions,
                "Direction": [
                    "Increases Risk" if c > 0 else "Decreases Risk"
                    for c in contributions
                ],
                "Abs_Contribution": np.abs(contributions),
            }
        )
        .sort_values("Abs_Contribution", ascending=False)
        .reset_index(drop=True)
    )

    return explanation_df


# =============================================================================
# Natural language explanation
# =============================================================================
def generate_natural_language_explanation(
    explanation_df: pd.DataFrame, risk_level: str, top_n: int = 5
) -> str:
    """Generate natural-language explanation from SHAP values."""
    if explanation_df.empty:
        return "Explanation not available."

    top_features = explanation_df.head(top_n)
    increasing = top_features[top_features["SHAP_Contribution"] > 0]
    decreasing = top_features[top_features["SHAP_Contribution"] < 0]

    text = f"**Prediction: {risk_level} Risk**\n\n"
    text += "**Features that contributed to higher risk:**\n"

    if len(increasing) > 0:
        for _, row in increasing.iterrows():
            text += (
                f"- **{row['Feature']}** = {row['Actual_Value']:.2f} "
                f"(contribution: +{row['SHAP_Contribution']:.4f})\n"
            )
    else:
        text += "- No features pushed risk upward significantly.\n"

    text += "\n**Features that contributed to lower risk:**\n"

    if len(decreasing) > 0:
        for _, row in decreasing.iterrows():
            text += (
                f"- **{row['Feature']}** = {row['Actual_Value']:.2f} "
                f"(contribution: {row['SHAP_Contribution']:.4f})\n"
            )
    else:
        text += "- No features pushed risk downward significantly.\n"

    text += "\n*Note: These features contributed to the model's prediction. "
    text += "This does not imply causality.*"

    return text


# =============================================================================
# Chat logic
# =============================================================================
def answer_explainability_chat(
    question: str,
    explanation_df: pd.DataFrame,
    probability: Optional[float] = None,
    global_importance: Optional[pd.DataFrame] = None,
) -> str:
    """Answer explainability questions or retrieve relevant project sources."""
    if explanation_df is None or explanation_df.empty:
        return (
            "I don't have an explanation loaded yet. Load a dataset and "
            "select a sample first."
        )

    normalized_question = question.lower().strip()
    positive = explanation_df[explanation_df["SHAP_Contribution"] > 0].head(3)
    negative = explanation_df[explanation_df["SHAP_Contribution"] < 0].head(3)

    def describe(rows: pd.DataFrame) -> str:
        if rows.empty:
            return "no features in this explanation"
        return ", ".join(
            f"**{row['Feature']}** ({row['SHAP_Contribution']:+.3f})"
            for _, row in rows.iterrows()
        )

    if any(term in normalized_question for term in ("cause", "caused", "causal")):
        return (
            "No. SHAP shows how the model used each feature; it does not show "
            "that a feature caused the risk. Explanations describe model behavior, "
            "not real-world causation."
        )

    if "global" in normalized_question or "overall" in normalized_question:
        if global_importance is not None and not global_importance.empty:
            top_features = global_importance.head(5)
            summary = ", ".join(
                f"**{row['Feature']}** ({row['Importance']:.3f})"
                for _, row in top_features.iterrows()
            )
            return (
                f"The strongest global contributors by mean absolute SHAP value "
                f"are: {summary}.\n\n"
                "Global importance summarizes **magnitude** across samples, "
                "not contribution direction for this single record."
            )
        return (
            "Global importance summarizes average absolute SHAP contributions "
            "across the analyzed samples."
        )

    if any(
        term in normalized_question
        for term in ("probability", "confidence", "score")
    ):
        if probability is None:
            return "A probability score is not available for this sample."
        return (
            f"The model reports **{probability:.1%}** for this sample.\n\n"
            "This is a model estimate, not a guarantee. It should not be treated "
            "as calibrated without a separate calibration evaluation."
        )

    if any(
        term in normalized_question
        for term in ("increase", "higher", "raise", "upward", "positive")
    ):
        return f"The strongest upward contributions are: {describe(positive)}."

    if any(
        term in normalized_question
        for term in ("decrease", "lower", "reduce", "downward", "negative")
    ):
        return f"The strongest downward contributions are: {describe(negative)}."

    if any(
        term in normalized_question
        for term in ("shap", "contribution", "mean", "value", "sign")
    ):
        return (
            "A **SHAP contribution** describes how a feature shifts the model "
            "output relative to its baseline.\n\n"
            "- The **sign** shows direction (+ raises risk, − lowers risk)\n"
            "- The **magnitude** shows strength of influence\n"
            "- Units depend on the model and explainer — they are **not** "
            "automatically percentage points"
        )

    project_terms = (
        "project", "module", "code", "architecture", "dataset", "preprocess",
        "forecast", "stockout", "stock-out", "supplier", "warehouse", "anomaly",
        "sustainab", "experiment", "application", "app", "algorithm",
        "technology", "train", "training",
    )
    if any(term in normalized_question for term in project_terms):
        try:
            from utils.project_knowledge import search_project_knowledge
            project_answer = search_project_knowledge(question)
            if project_answer:
                return project_answer
        except Exception:
            pass

    if any(
        term in normalized_question
        for term in ("change", "alter", "modify", "affect the prediction")
    ):
        return (
            "No. This chat only **interprets** the displayed results. It does not "
            "change the model, inputs, prediction, or SHAP values."
        )

    if any(
        term in normalized_question
        for term in ("why", "explain", "prediction", "risk")
    ):
        return (
            f"For this sample:\n\n"
            f"- **Upward contributions:** {describe(positive)}\n"
            f"- **Downward contributions:** {describe(negative)}\n\n"
            "These explain the model output — not real-world causation."
        )

    try:
        from utils.project_knowledge import search_project_knowledge
        project_answer = search_project_knowledge(question)
        if project_answer:
            return project_answer
    except Exception:
        pass

    return (
        "I couldn't find a relevant answer in the project files.\n\n"
        "**Try asking about:**\n"
        "- A project module (forecasting, stock-out, supplier risk, anomaly detection)\n"
        "- SHAP values or feature contributions\n"
        "- The dataset, preprocessing, or model training"
    )


# =============================================================================
# The 15 questions — always visible to the user
# =============================================================================
QUESTION_POOL: List[str] = [
    "Why was this sample flagged as high risk?",
    "Which features pushed the risk upward?",
    "Which features lowered the risk?",
    "What is the probability score for this sample?",
    "What is a SHAP contribution?",
    "How do I read the local explanation plot?",
    "What is the difference between global and local explanations?",
    "Why can't SHAP prove causation?",
    "Explain the demand forecasting module",
    "How does the stock-out warning system work?",
    "How is supplier risk scored?",
    "What does the anomaly detection module do?",
    "Which ML models are used in SupplyGuard?",
    "How is the data preprocessed?",
    "What technology stack powers this app?",
]


def render_all_question_chips(key_prefix: str = "q") -> Optional[str]:
    """
    Render ALL 15 questions as clickable chips inside an expander.
    Returns the clicked question, or None.
    """
    clicked: Optional[str] = None

    with st.expander("💬 Ask a question (click to expand)", expanded=True):
        st.caption("Pick any question below, or type your own at the bottom.")

        cols = st.columns(3)
        for i, question in enumerate(QUESTION_POOL):
            col = cols[i % 3]
            with col:
                if st.button(
                    question,
                    key=f"{key_prefix}_{i}",
                    use_container_width=True,
                ):
                    clicked = question

    return clicked


# =============================================================================
# Chat UI
# =============================================================================
def render_explainability_chat(
    explanation_df: Optional[pd.DataFrame] = None,
    probability: Optional[float] = None,
    global_importance: Optional[pd.DataFrame] = None,
    dataset_loaded: bool = False,
) -> None:
    """
    Render the Explainability Chat UI with all 15 questions always visible.
    Answers are grounded in project files and do not modify predictions.
    """
    st.markdown("### 🧠 Explainability Chat")
    st.caption(
        "Ask about this prediction or any SupplyGuard module. "
        "Answers are grounded in project files and **do not modify predictions**."
    )

    # --- Session state init ---
    if "explain_chat_history" not in st.session_state:
        welcome = (
            "👋 Hi! I can help you understand this prediction, explain "
            "SHAP values, or look up how any SupplyGuard module works."
        )
        if not dataset_loaded:
            welcome += (
                "\n\n⚠️ **No prediction loaded yet.** You can still ask about "
                "project modules, models, or datasets.\n\n"
                "To unlock prediction-specific answers, load a dataset and "
                "train the models first."
            )
        welcome += "\n\n👇 **Pick a question below or type your own.**"
        st.session_state.explain_chat_history = [
            {"role": "assistant", "content": welcome}
        ]

    # --- Render chat history ---
    for message in st.session_state.explain_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ALWAYS show all 15 questions ---
    user_question: Optional[str] = render_all_question_chips(key_prefix="q")

    # --- Chat input ---
    typed_question = st.chat_input("Or type your own question...")
    if typed_question:
        user_question = typed_question

    # --- Handle question ---
    if user_question:
        st.session_state.explain_chat_history.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if explanation_df is None or explanation_df.empty:
                    try:
                        from utils.project_knowledge import search_project_knowledge
                        answer = search_project_knowledge(user_question)
                        if not answer:
                            answer = (
                                "I can only answer **project-related** questions "
                                "right now because no prediction is loaded.\n\n"
                                "**Load a dataset** and train the models to unlock:\n"
                                "- *Why was this sample flagged?*\n"
                                "- *Which features pushed risk upward?*\n"
                                "- *What is the probability for this sample?*"
                            )
                    except Exception:
                        answer = (
                            "Project knowledge module not found. Load a dataset "
                            "to unlock prediction-specific answers."
                        )
                else:
                    answer = answer_explainability_chat(
                        question=user_question,
                        explanation_df=explanation_df,
                        probability=probability,
                        global_importance=global_importance,
                    )
            st.markdown(answer)

        st.session_state.explain_chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    # --- Footer ---
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(
            "🔒 **Note:** This chat interprets the displayed results only. "
            "It does not alter the model, inputs, predictions, or SHAP values. "
            "Explanations describe model behavior, not real-world causality."
        )
    with col2:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.explain_chat_history = [
                st.session_state.explain_chat_history[0]
            ]
            st.rerun()


# =============================================================================
# Plotting
# =============================================================================
def plot_local_explanation(
    explanation_df: pd.DataFrame, top_n: int = 10
) -> "go.Figure":
    """Create a waterfall-style plot for local explanation."""
    import plotly.graph_objects as go

    top = explanation_df.head(top_n).sort_values("SHAP_Contribution")

    colors = [
        "#e74c3c" if c > 0 else "#2ecc71" for c in top["SHAP_Contribution"]
    ]

    fig = go.Figure(
        go.Bar(
            x=top["SHAP_Contribution"],
            y=top["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{c:+.4f}" for c in top["SHAP_Contribution"]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Local Feature Contributions (SHAP)",
        xaxis_title="SHAP Contribution",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, len(top) * 30),
    )

    return fig