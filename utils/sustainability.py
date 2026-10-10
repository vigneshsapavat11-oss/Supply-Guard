"""
SupplyGuard - Sustainability Module
Computes RELATIVE sustainability indicators.
NOT scientifically validated environmental metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from config import SUSTAINABILITY_WEIGHTS


def compute_inventory_waste_risk(df: pd.DataFrame,
                                  inventory_col: str = "inventory",
                                  demand_col: str = "demand") -> float:
    """
    Compute relative inventory waste risk.
    Higher excess inventory relative to demand = higher waste risk.
    Returns value between 0 and 1.
    """
    if inventory_col not in df.columns or demand_col not in df.columns:
        return 0.0
    
    ratio = df[inventory_col] / (df[demand_col] + 1)
    # Normalize: ratios above 3 are considered high waste risk
    waste_risk = np.clip(ratio / 3.0, 0, 1).mean()
    return round(float(waste_risk), 4)


def compute_transportation_burden(transfer_df: pd.DataFrame) -> float:
    """
    Compute relative transportation burden based on redistribution volume.
    """
    if transfer_df is None or len(transfer_df) == 0:
        return 0.0
    
    if "transfer_quantity" not in transfer_df.columns:
        return 0.0
    
    total_transfer = transfer_df["transfer_quantity"].sum()
    # Normalize using a reference value (configurable)
    reference = 10000
    burden = np.clip(total_transfer / reference, 0, 1)
    return round(float(burden), 4)


def compute_emergency_replenishment_burden(df: pd.DataFrame,
                                            days_of_inventory_col: str = "days_of_inventory") -> float:
    """
    Compute relative emergency replenishment burden.
    More products with critically low inventory = higher burden.
    """
    if days_of_inventory_col not in df.columns:
        return 0.0
    
    critical = (df[days_of_inventory_col] < 3).sum()
    total = len(df)
    
    if total == 0:
        return 0.0
    
    burden = np.clip(critical / total * 2, 0, 1)  # Scale up
    return round(float(burden), 4)


def compute_redistribution_efficiency(transfer_df: pd.DataFrame,
                                       shortage_before: float = None,
                                       shortage_after: float = None) -> float:
    """
    Compute redistribution efficiency.
    Higher = better (more shortage resolved).
    """
    if shortage_before is not None and shortage_after is not None:
        if shortage_before > 0:
            efficiency = 1 - (shortage_after / shortage_before)
            return round(float(np.clip(efficiency, 0, 1)), 4)
    
    # Fallback: based on whether transfers were recommended
    if transfer_df is None or len(transfer_df) == 0:
        return 0.0
    
    return 0.5  # Neutral when we can't compute


def compute_sustainability_indicators(df: pd.DataFrame,
                                       transfer_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Compute all relative sustainability indicators.
    These are RELATIVE indicators, not absolute environmental measurements.
    """
    indicators = {
        "Inventory Waste Risk": compute_inventory_waste_risk(df),
        "Transportation Burden": compute_transportation_burden(transfer_df) if transfer_df is not None else 0.0,
        "Emergency Replenishment Burden": compute_emergency_replenishment_burden(df),
        "Redistribution Efficiency": compute_redistribution_efficiency(transfer_df)
    }
    
    # Compute weighted composite score
    composite = sum(indicators[k] * SUSTAINABILITY_WEIGHTS.get(k.lower().replace(" ", "_"), 0.25)
                   for k in indicators)
    indicators["Composite Sustainability Score"] = round(composite, 4)
    
    return indicators


def get_sustainability_recommendations(indicators: Dict[str, float]) -> list:
    """Generate recommendations based on sustainability indicators."""
    recommendations = []
    
    if indicators.get("Inventory Waste Risk", 0) > 0.6:
        recommendations.append(
            "⚠️ High inventory waste risk detected. Consider reducing excess inventory "
            "through redistribution or demand-based ordering."
        )
    
    if indicators.get("Transportation Burden", 0) > 0.6:
        recommendations.append(
            "⚠️ High transportation burden. Consider consolidating shipments or "
            "sourcing from closer warehouses."
        )
    
    if indicators.get("Emergency Replenishment Burden", 0) > 0.5:
        recommendations.append(
            "⚠️ Many products at critically low inventory. Review safety stock levels "
            "and lead times to reduce emergency orders."
        )
    
    if indicators.get("Redistribution Efficiency", 0) < 0.3 and indicators.get("Redistribution Efficiency", 0) > 0:
        recommendations.append(
            "⚠️ Low redistribution efficiency. Review transfer recommendations and "
            "warehouse network configuration."
        )
    
    if not recommendations:
        recommendations.append("✅ Sustainability indicators are within acceptable ranges.")
    
    return recommendations