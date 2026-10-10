"""SupplyGuard - Research Experiments Page"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import platform
from datetime import datetime
from utils.ui_theme import apply_enterprise_theme
from utils.research import dataset_fingerprint, make_experiment_split, prepare_experiment_features
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
import sklearn
from utils.metrics import regression_metrics, classification_metrics

st.set_page_config(page_title="Research Experiments", page_icon="🔬", layout="wide")
apply_enterprise_theme()

st.markdown("# 🔬 Research Experiments")
st.markdown("Run controlled experiments and store results for research reproducibility.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df

# Experiment log
if "experiment_log" not in st.session_state:
    st.session_state.experiment_log = []

# Experiment configuration
st.markdown("### ⚙️ Experiment Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    experiment_name = st.text_input("Experiment Name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model_type = st.selectbox("Model Type", ["Random Forest", "Linear Regression", "Logistic Regression"])

with col2:
    if model_type == "Random Forest":
        n_estimators = st.number_input("n_estimators", 10, 500, 100, 10)
        max_depth = st.number_input("max_depth", 2, 30, 10, 1)
    else:
        n_estimators = None
        max_depth = None
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

with col3:
    if model_type == "Random Forest":
        min_samples_split = st.number_input("min_samples_split", 2, 20, 5, 1)
    else:
        min_samples_split = None
    random_state = st.number_input("Random State", 0, 100, 42, 1)

# Target selection
st.markdown("### 🎯 Target Variable")
numeric_targets = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
if not numeric_targets:
    st.warning("The selected dataset has no numeric target column for an experiment.")
    st.stop()

target_col = st.selectbox("Target", numeric_targets)
experiment_features = prepare_experiment_features(df, target_col)

if target_col.lower() not in {"demand", "sales", "quantity", "units_sold"}:
    st.info("This experiment evaluates the selected numeric target (for example, food price). It does not create inventory, stock-out, or supplier data.")

# Feature selection
st.markdown("### 📋 Feature Selection")
available_features = experiment_features.columns.tolist()

selected_features = st.multiselect("Features", available_features, 
                                   default=available_features[:min(10, len(available_features))])

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# Run experiment
if st.button("🚀 Run Experiment", type="primary"):
    with st.spinner("Running experiment..."):
        try:
            X = experiment_features[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = df[target_col]
            
            # Remove NaN targets
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            
            is_regression = y.dtype in [np.float64, np.float32] and len(y.unique()) > 10
            task = "regression" if is_regression else "classification"
            date_columns = [
                column for column in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[column])
            ]
            experiment_dates = df.loc[mask, date_columns[0]] if date_columns else None
            train_indices, test_indices, split_method = make_experiment_split(
                len(y), test_size, random_state, dates=experiment_dates,
                stratify=None if is_regression else y,
            )
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            start_time = datetime.now()
            
            if model_type == "Random Forest":
                if is_regression:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_split=min_samples_split, random_state=random_state, n_jobs=-1
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_split=min_samples_split, random_state=random_state, n_jobs=-1
                    )
            elif model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = LogisticRegression(max_iter=1000, random_state=random_state)
            
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            y_pred = model.predict(X_test)
            baseline_model = (
                DummyRegressor(strategy="mean") if is_regression
                else DummyClassifier(strategy="most_frequent")
            )
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_test)
            
            if task == "regression":
                metrics = regression_metrics(y_test, y_pred)
                baseline_metrics = regression_metrics(y_test, baseline_pred)
            else:
                metrics = classification_metrics(y_test, y_pred)
                baseline_metrics = classification_metrics(y_test, baseline_pred)
            
            # Log experiment
            experiment = {
                "experiment_id": experiment_name,
                "timestamp": datetime.now().astimezone().isoformat(),
                "dataset_size": len(df),
                "dataset_sha256": dataset_fingerprint(df),
                "target": target_col,
                "feature_count": len(selected_features),
                "features": selected_features,
                "model": model_type,
                "task": task,
                "split_method": split_method,
                "test_rows": len(test_indices),
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "hyperparameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "test_size": test_size,
                    "random_state": random_state
                },
                "metrics": metrics,
                "baseline": "mean" if is_regression else "most_frequent",
                "baseline_metrics": baseline_metrics,
                "training_time": round(training_time, 3)
            }
            
            st.session_state.experiment_log.append(experiment)
            
            st.success("✅ Experiment completed!")
            
            # Display results
            st.markdown("### 📊 Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Training Time", f"{training_time:.2f}s")
            col2.metric("Test Size", f"{len(X_test)}")
            col3.metric("Features", len(selected_features))
            
            if task == "regression":
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
                col2.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
                col3.metric("R²", f"{metrics.get('R2', 0):.4f}")
                col4.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
                col2.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
                col3.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
                col4.metric("F1", f"{metrics.get('F1', 0):.4f}")

            st.markdown("#### Model vs. naive baseline")
            comparison = {
                name: {"Model": value, "Baseline": baseline_metrics.get(name)}
                for name, value in metrics.items()
                if np.isscalar(value) and name in baseline_metrics
            }
            st.dataframe(pd.DataFrame.from_dict(comparison, orient="index"), use_container_width=True)
            
        except Exception as e:
            st.error(f"Experiment failed: {e}")

# Experiment history
st.markdown("---")
st.markdown("### 📋 Experiment History")

if len(st.session_state.experiment_log) > 0:
    history_df = pd.DataFrame(st.session_state.experiment_log)
    
    # Flatten metrics
    display_df = history_df[["experiment_id", "timestamp", "model", "task", 
                             "dataset_size", "feature_count", "training_time"]].copy()
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download
    csv = history_df.to_json(orient="records", indent=2)
    st.download_button("📥 Download Experiment Log (JSON)", csv,
                      "experiment_log.json", "application/json")
    
    # Detailed view
    with st.expander("View Full Experiment Details"):
        for exp in st.session_state.experiment_log:
            st.markdown(f"**{exp['experiment_id']}**")
            st.json(exp)
            st.markdown("---")
else:
    st.info("No experiments run yet. Configure and run an experiment above.")

# Ablation study
st.markdown("---")
st.markdown("### 🔬 Ablation Study")
st.markdown("Compare model performance with different feature groups.")

if st.button("Run Ablation Study"):
    st.info("Ablation studies are not yet implemented. Use separate runs with explicitly recorded feature sets; "
            "do not interpret the current placeholder as an ablation result.")