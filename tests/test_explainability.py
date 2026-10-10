"""Tests for SHAP output normalization and feature importance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.explainability import (
    _select_output_values,
    answer_explainability_chat,
    compute_shap_values,
    explain_local_prediction,
    get_global_feature_importance,
)


def test_interventional_shap_explains_fitted_tree_pipeline():
    features = pd.DataFrame({
        "inventory": np.linspace(10, 100, 40),
        "lead_time": np.tile([2.0, 8.0], 20),
    })
    labels = (features["inventory"] < 50).astype(int)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])
    model.fit(features, labels)

    shap_values = compute_shap_values(
        model, features.iloc[:8], features.columns.tolist(), model_type="tree"
    )

    assert shap_values.shape == (8, 2)
    assert np.isfinite(shap_values).all()


def test_global_importance_reduces_multiclass_shap_values():
    """Aggregate sample and output dimensions into one score per feature."""
    shap_values = np.array([
        [[1.0, -3.0], [2.0, -4.0]],
        [[3.0, -1.0], [4.0, -2.0]],
    ])

    importance = get_global_feature_importance(shap_values, ["inventory", "demand"])

    assert importance["Feature"].tolist() == ["demand", "inventory"]
    assert importance["Importance"].tolist() == [3.0, 2.0]


def test_global_importance_keeps_two_dimensional_behavior():
    """Preserve the sample-by-feature aggregation used by regression SHAP."""
    shap_values = np.array([[1.0, -2.0], [-3.0, 4.0]])

    importance = get_global_feature_importance(shap_values, ["inventory", "demand"])

    assert importance["Feature"].tolist() == ["demand", "inventory"]
    assert importance["Importance"].tolist() == [3.0, 2.0]


def test_local_explanation_selects_positive_class_values():
    """Use the positive-class output for one local classifier explanation."""
    shap_values = np.array([
        [[0.1, -0.2], [0.3, -0.4]],
        [[0.5, -0.6], [0.7, -0.8]],
    ])
    features = pd.DataFrame([[10, 20], [30, 40]], columns=["inventory", "demand"])

    local = explain_local_prediction(shap_values, features, ["inventory", "demand"], 1)

    contributions = local.set_index("Feature")["SHAP_Contribution"]
    assert contributions["inventory"] == -0.6
    assert contributions["demand"] == -0.8


def test_output_normalization_selects_positive_class():
    shap_values = np.arange(24).reshape(3, 4, 2)

    normalized = _select_output_values(shap_values)

    assert normalized.shape == (3, 4)
    np.testing.assert_array_equal(normalized, shap_values[:, :, 1])


def test_explainability_chat_answers_about_local_risk_drivers():
    local = pd.DataFrame({
        "Feature": ["inventory", "lead_time"],
        "SHAP_Contribution": [0.25, -0.1],
    })

    answer = answer_explainability_chat("Why is the risk high?", local)

    assert "inventory (+0.250)" in answer
    assert "lead_time (-0.100)" in answer


def test_explainability_chat_rejects_causal_interpretation():
    local = pd.DataFrame({"Feature": ["inventory"], "SHAP_Contribution": [0.25]})

    answer = answer_explainability_chat("Did inventory cause the risk?", local)

    assert "does not show that a feature caused the risk" in answer


def test_explainability_chat_reports_read_only_probability():
    local = pd.DataFrame({"Feature": ["inventory"], "SHAP_Contribution": [0.25]})

    answer = answer_explainability_chat("What is the probability?", local, 0.8)

    assert "80.0%" in answer
    assert "not a guarantee" in answer
