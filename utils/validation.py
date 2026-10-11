"""
SupplyGuard - Schema Validation
Validates that a DataFrame has the columns needed by each module.

Provides:
    - validate_schema(df)        -> returns a report dict
    - raise_if_invalid(report)   -> raises ValueError if required columns are missing
    - get_schema_summary(report) -> human-readable summary string (optional helper)
"""

from typing import Any

import pandas as pd

from config import EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Column requirements
# ---------------------------------------------------------------------------
# Hard-required: the app cannot do anything meaningful without these
REQUIRED_COLUMNS = ["demand"]

# Soft-recommended: warn the user if missing, but do not block
RECOMMENDED_COLUMNS = [
    "date",
    "product_id",
    "warehouse",
    "supplier_id",
    "inventory",
]


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate a DataFrame against SupplyGuard's expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        The processed dataset.

    Returns
    -------
    dict
        {
            "valid": bool,
            "missing_required": list[str],
            "missing_recommended": list[str],
            "present_columns": list[str],
            "extra_columns": list[str],
            "row_count": int,
            "column_count": int,
            "error": str | None,
        }
    """
    # -- Handle None or empty ------------------------------------------------
    if df is None:
        return {
            "valid": False,
            "missing_required": REQUIRED_COLUMNS,
            "missing_recommended": RECOMMENDED_COLUMNS,
            "present_columns": [],
            "extra_columns": [],
            "row_count": 0,
            "column_count": 0,
            "error": "DataFrame is None.",
        }

    if not isinstance(df, pd.DataFrame):
        return {
            "valid": False,
            "missing_required": REQUIRED_COLUMNS,
            "missing_recommended": RECOMMENDED_COLUMNS,
            "present_columns": [],
            "extra_columns": [],
            "row_count": 0,
            "column_count": 0,
            "error": f"Expected pandas.DataFrame, got {type(df).__name__}.",
        }

    if df.empty:
        return {
            "valid": False,
            "missing_required": REQUIRED_COLUMNS,
            "missing_recommended": RECOMMENDED_COLUMNS,
            "present_columns": list(df.columns),
            "extra_columns": [],
            "row_count": 0,
            "column_count": len(df.columns),
            "error": "DataFrame is empty.",
        }

    # -- Compare columns -----------------------------------------------------
    present = list(df.columns)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in present]
    missing_recommended = [c for c in RECOMMENDED_COLUMNS if c not in present]

    expected = set(EXPECTED_COLUMNS)
    extra = [c for c in present if c not in expected]

    return {
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "present_columns": present,
        "extra_columns": extra,
        "row_count": len(df),
        "column_count": len(df.columns),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Raise on invalid
# ---------------------------------------------------------------------------
def raise_if_invalid(report: dict) -> None:
    """
    Raise ValueError if the schema report indicates the dataset is invalid.

    Only blocks on missing REQUIRED columns -- missing recommended columns
    are treated as warnings by the caller, not errors.
    """
    if not isinstance(report, dict):
        raise ValueError("Schema report must be a dict.")

    # Blocking error (None / not a DataFrame / empty)
    if report.get("error") and not report.get("present_columns"):
        raise ValueError(report["error"])

    # Blocking error (missing required columns)
    if not report.get("valid", False):
        missing = report.get("missing_required", [])
        if missing:
            raise ValueError(
                f"Dataset is missing required column(s): {missing}. "
                f"Please add them and try again."
            )
        raise ValueError("Dataset failed schema validation.")


# ---------------------------------------------------------------------------
# Optional helper for UI display
# ---------------------------------------------------------------------------
def get_schema_summary(report: dict) -> str:
    """Return a short human-readable summary of a schema report."""
    if not isinstance(report, dict):
        return "Invalid schema report."

    parts = [
        f"Rows: {report.get('row_count', 0):,}",
        f"Columns: {report.get('column_count', 0)}",
        f"Valid: {report.get('valid', False)}",
    ]

    if report.get("missing_required"):
        parts.append(f"Missing required: {report['missing_required']}")
    if report.get("missing_recommended"):
        parts.append(f"Missing recommended: {report['missing_recommended']}")

    return " | ".join(parts)