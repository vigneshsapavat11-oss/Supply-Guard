"""SupplyGuard - Explainable AI Page"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager
from utils.explainability import (
    compute_shap_values, get_global_feature_importance,
    explain_local_prediction, generate_natural_language_explanation,
    plot_local_explanation, check_shap_available, answer_explainability_chat
)

st.set_page_config(page_title="Explainable AI", page_icon="🧠", layout="wide")
apply_enterprise_theme()

st.markdown("# 🧠 Explainable AI")
st.markdown("SHAP-based explanations for model predictions.")

if not check_shap_available():
    st.warning("⚠️ SHAP library is not installed. Install with: `pip install shap`")
    st.stop()

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
manager = get_model_manager()

# Model selection
st.markdown("### 🎯 Select Model to Explain")
model_choice = st.selectbox("Model", ["Stock-Out Risk", "Supplier Risk"])

if model_choice == "Stock-Out Risk":
    predictor = manager.stockout_predictor
    if predictor is None or predictor.model is None:
        st.warning("⚠️ Stock-out model not trained.")
        st.stop()
    model = predictor.model
    feature_names = predictor.feature_names
    task = "stockout"
else:
    predictor = manager.supplier_predictor
    if predictor is None or predictor.model is None:
        st.warning("⚠️ Supplier risk model not trained.")
        st.stop()
    model = predictor.model
    feature_names = predictor.feature_names
    task = "supplier"

# Prepare data
if task == "stockout":
    X, _ = predictor.prepare_features(df)
    X = X.reindex(columns=feature_names).replace([np.inf, -np.inf], np.nan)
else:
    supplier_df = predictor.supplier_features
    X = supplier_df.reindex(columns=feature_names).replace([np.inf, -np.inf], np.nan)

# Compute SHAP values
st.markdown("### 📊 Computing SHAP Values")
st.caption("This may take a moment for large datasets.")

if st.button("Compute SHAP Explanations"):
    with st.spinner("Computing SHAP values..."):
        # Sample for performance
        sample_size = min(200, len(X))
        X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        
        model_type = "tree" if hasattr(model, "named_steps") and \
                    any("Forest" in str(type(model.named_steps.get("model", None))) or
                        "XGB" in str(type(model.named_steps.get("model", None))) or
                        "Gradient" in str(type(model.named_steps.get("model", None))) or
                        "LGBM" in str(type(model.named_steps.get("model", None)))
                        for _ in [1]) else "other"

        shap_values = compute_shap_values(model, X_sample, feature_names, model_type)
        
        if shap_values is not None:
            st.session_state[f"shap_values_{task}"] = shap_values
            st.session_state[f"shap_X_{task}"] = X_sample
            chat_prefix = f"explainability_chat_{task}_"
            for state_key in list(st.session_state):
                if state_key.startswith(chat_prefix):
                    del st.session_state[state_key]
            st.success(f"✅ SHAP values computed for {len(X_sample)} samples")
        else:
            st.error("SHAP computation failed.")

# Display global importance
if f"shap_values_{task}" in st.session_state:
    shap_values = st.session_state[f"shap_values_{task}"]
    X_sample = st.session_state[f"shap_X_{task}"]
    
    st.markdown("---")
    st.markdown("### 🌍 Global Feature Importance")
    
    importance_df = get_global_feature_importance(shap_values, feature_names)
    
    import plotly.express as px
    fig = px.bar(importance_df.head(20), x="Importance", y="Feature",
                 orientation="h", title="Mean |SHAP| Feature Importance")
    fig.update_layout(template="plotly_white", height=500)
    render_plotly(fig, use_container_width=True)
    
    st.dataframe(importance_df, use_container_width=True)
    
    # Local explanation
    st.markdown("---")
    st.markdown("### 🔍 Local Explanation")
    
    idx = st.number_input("Select sample index", 0, len(X_sample) - 1, 0)
    
    local_df = explain_local_prediction(shap_values, X_sample, feature_names, idx)
    
    if not local_df.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = plot_local_explanation(local_df, top_n=10)
            render_plotly(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Feature Contributions")
            st.dataframe(
                local_df[["Feature", "Actual_Value", "SHAP_Contribution", "Direction"]].head(10),
                use_container_width=True
            )
        
        # Natural language explanation
        st.markdown("### 📝 Natural Language Explanation")
        
        # Determine risk level
        if task == "stockout":
            prob = predictor.model.predict_proba(X_sample.iloc[[idx]])[0, 1]
            risk_level = "HIGH" if prob > 0.5 else "LOW"
        else:
            prob = predictor.model.predict_proba(X_sample.iloc[[idx]])[0, 1]
            risk_level = "HIGH" if prob > 0.5 else "LOW"
        
        explanation = generate_natural_language_explanation(local_df, risk_level)
        st.markdown(explanation)

        st.markdown("### 💬 Explainability Chat")
        st.caption("Ask about this prediction or any SupplyGuard module. Answers are grounded in project files and do not change predictions.")

        chat_key = f"explainability_chat_{task}_{int(idx)}"
        chat_history = st.session_state.setdefault(chat_key, [])
        if not chat_history:
            with st.chat_message("assistant"):
                st.markdown("I can explain this sample or look up how a SupplyGuard feature works. Ask about forecasting, risk models, data, SHAP, or another project topic.")

        for message in chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about this explanation", key=f"chat_input_{task}_{int(idx)}"):
            chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            answer = answer_explainability_chat(
                prompt, local_df, prob, global_importance=importance_df
            )
            chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        if chat_history and st.button("Clear conversation", key=f"clear_chat_{task}_{int(idx)}"):
            st.session_state[chat_key] = []
            st.rerun()
