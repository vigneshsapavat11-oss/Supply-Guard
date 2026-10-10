"""Tests for project-wide source retrieval used by the Explainable AI chat."""

import pandas as pd

from utils.project_knowledge import search_project_knowledge
from utils.explainability import answer_explainability_chat


def test_search_returns_project_sources_for_module_questions():
    answer = search_project_knowledge("Which models are used for demand forecasting?")

    assert answer is not None
    assert "Linear Regression" in answer
    assert "demand_forecasting.py" in answer or "README.md" in answer
    assert "```" not in answer
    assert len(answer.splitlines()) <= 3


def test_search_returns_no_answer_for_unrelated_question():
    answer = search_project_knowledge("What is the recipe for chocolate cake?")

    assert answer is None


def test_chat_routes_project_questions_to_source_retrieval():
    local = pd.DataFrame({"Feature": ["inventory"], "SHAP_Contribution": [0.2]})

    answer = answer_explainability_chat(
        "Which models are used for demand forecasting?", local
    )

    assert "Linear Regression" in answer
    assert "demand_forecasting.py" in answer or "README.md" in answer