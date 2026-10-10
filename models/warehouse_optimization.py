"""
SupplyGuard - Multi-Warehouse Redistribution Optimization
Identifies surplus/shortage and recommends transfers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class WarehouseOptimizer:
    """
    Analyzes inventory across warehouses and recommends transfers.
    Uses a transparent, rule-based optimization approach.
    """
    
    def __init__(self, safety_stock_days: float = 3.0,
                 min_transfer_quantity: int = 10):
        self.safety_stock_days = safety_stock_days
        self.min_transfer_quantity = min_transfer_quantity
        self.recommendations: Optional[pd.DataFrame] = None
        self.summary: Dict = {}
    
    def analyze_warehouse_inventory(self, df: pd.DataFrame,
                                     demand_col: str = "demand",
                                     product_col: str = "product_id",
                                     warehouse_col: str = "warehouse") -> pd.DataFrame:
        """
        Analyze inventory levels per product-warehouse.
        """
        # Aggregate recent data per product-warehouse
        agg = df.groupby([product_col, warehouse_col]).agg(
            avg_demand=(demand_col, "mean"),
            demand_std=(demand_col, "std"),
            current_inventory=("inventory", "last"),
            avg_lead_time=("lead_time", "mean") if "lead_time" in df.columns else (demand_col, "count")
        ).reset_index()
        
        # Fill NaN
        agg["demand_std"] = agg["demand_std"].fillna(0)
        
        # Calculate days of inventory
        agg["days_of_inventory"] = np.where(
            agg["avg_demand"] > 0,
            agg["current_inventory"] / agg["avg_demand"],
            999
        )
        
        # Safety stock
        agg["safety_stock"] = np.maximum(
            self.safety_stock_days * agg["avg_demand"],
            0
        )
        
        # Optimal inventory (target)
        agg["target_inventory"] = agg["avg_demand"] * (self.safety_stock_days + 
                                                        agg.get("avg_lead_time", 0))
        
        # Surplus / Shortage
        agg["surplus"] = np.maximum(0, agg["current_inventory"] - agg["target_inventory"] * 1.2)
        agg["shortage"] = np.maximum(0, agg["target_inventory"] - agg["current_inventory"])
        
        # Status
        agg["status"] = "HEALTHY"
        agg.loc[agg["days_of_inventory"] < 2, "status"] = "CRITICAL"
        agg.loc[(agg["days_of_inventory"] >= 2) & (agg["days_of_inventory"] < 5), "status"] = "LOW"
        agg.loc[agg["days_of_inventory"] > 15, "status"] = "OVERSTOCK"
        
        return agg
    
    def generate_transfer_recommendations(self, df: pd.DataFrame,
                                           max_transfers: int = 50) -> pd.DataFrame:
        """
        Generate transfer recommendations between warehouses.
        For each product, match surplus warehouses to shortage warehouses.
        """
        analysis = self.analyze_warehouse_inventory(df)
        
        recommendations = []
        
        for product in analysis["product_id"].unique():
            product_data = analysis[analysis["product_id"] == product]
            
            # Source warehouses (have surplus)
            sources = product_data[product_data["surplus"] >= self.min_transfer_quantity].copy()
            sources = sources.sort_values("surplus", ascending=False)
            
            # Destination warehouses (have shortage)
            destinations = product_data[product_data["shortage"] > 0].copy()
            destinations = destinations.sort_values("shortage", ascending=False)
            
            if len(sources) == 0 or len(destinations) == 0:
                continue
            
            for _, dest in destinations.iterrows():
                remaining_shortage = dest["shortage"]
                
                for idx, src in sources.iterrows():
                    if remaining_shortage <= 0:
                        break
                    if src["surplus"] <= 0:
                        continue
                    
                    transfer_qty = min(src["surplus"], remaining_shortage)
                    
                    if transfer_qty >= self.min_transfer_quantity:
                        priority = "HIGH" if dest["status"] == "CRITICAL" else \
                                  "MEDIUM" if dest["status"] == "LOW" else "LOW"
                        
                        recommendations.append({
                            "product_id": product,
                            "source_warehouse": src["warehouse"],
                            "destination_warehouse": dest["warehouse"],
                            "transfer_quantity": int(transfer_qty),
                            "priority": priority,
                            "source_surplus_before": int(src["surplus"]),
                            "destination_shortage_before": int(dest["shortage"]),
                            "estimated_transport_units": int(transfer_qty)
                        })
                        
                        # Update remaining
                        sources.at[idx, "surplus"] -= transfer_qty
                        remaining_shortage -= transfer_qty
                    
                    if len(recommendations) >= max_transfers:
                        break
                
                if len(recommendations) >= max_transfers:
                    break
            
            if len(recommendations) >= max_transfers:
                break
        
        self.recommendations = pd.DataFrame(recommendations)
        
        # Compute summary
        if len(self.recommendations) > 0:
            self.summary = {
                "total_transfers": len(self.recommendations),
                "total_units": int(self.recommendations["transfer_quantity"].sum()),
                "products_affected": self.recommendations["product_id"].nunique(),
                "high_priority": (self.recommendations["priority"] == "HIGH").sum()
            }
        else:
            self.summary = {
                "total_transfers": 0,
                "total_units": 0,
                "products_affected": 0,
                "high_priority": 0
            }
        
        return self.recommendations
    
    def get_warehouse_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics per warehouse."""
        analysis = self.analyze_warehouse_inventory(df)
        
        summary = analysis.groupby("warehouse").agg(
            total_inventory=("current_inventory", "sum"),
            avg_days_of_inventory=("days_of_inventory", "mean"),
            critical_products=("status", lambda x: (x == "CRITICAL").sum()),
            low_stock_products=("status", lambda x: (x == "LOW").sum()),
            overstock_products=("status", lambda x: (x == "OVERSTOCK").sum()),
            total_surplus=("surplus", "sum"),
            total_shortage=("shortage", "sum")
        ).reset_index()
        
        return summary