"""SupplyGuard - Warehouse Optimization Page"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.ui_theme import apply_enterprise_theme, render_plotly
from models.model_manager import get_model_manager
from models.warehouse_optimization import WarehouseOptimizer

st.set_page_config(page_title="Warehouse Optimization", page_icon="📦", layout="wide")
apply_enterprise_theme()

st.markdown("# 📦 Multi-Warehouse Redistribution")
st.markdown("Optimize inventory across warehouses to reduce shortages and excess stock.")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please load a dataset first.")
    st.stop()

df = st.session_state.processed_df
required_columns = {"product_id", "warehouse", "demand", "inventory"}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    st.info(f"Warehouse optimization is unavailable for this dataset. Missing required columns: {', '.join(missing_columns)}.")
    st.stop()

# Initialize optimizer
if manager := get_model_manager():
    if manager.warehouse_optimizer is None:
        manager.warehouse_optimizer = WarehouseOptimizer()

optimizer = manager.warehouse_optimizer

# Configuration
st.markdown("### ⚙️ Configuration")
col1, col2 = st.columns(2)
with col1:
    safety_days = st.slider("Safety Stock (days)", 1.0, 10.0, 3.0, 0.5)
with col2:
    min_transfer = st.slider("Minimum Transfer Quantity", 5, 100, 10, 5)

optimizer.safety_stock_days = safety_days
optimizer.min_transfer_quantity = min_transfer

# Warehouse summary
st.markdown("### 📊 Warehouse Inventory Summary")
try:
    summary = optimizer.get_warehouse_summary(df)
    st.dataframe(summary, use_container_width=True)
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# Generate recommendations
st.markdown("### 🔄 Transfer Recommendations")

if st.button("Generate Transfer Recommendations", type="primary"):
    try:
        with st.spinner("Analyzing inventory imbalance..."):
            recommendations = optimizer.generate_transfer_recommendations(df)
        
        if len(recommendations) > 0:
            st.success(f"✅ Generated {len(recommendations)} transfer recommendations")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transfers", optimizer.summary["total_transfers"])
            col2.metric("Total Units", f"{optimizer.summary['total_units']:,}")
            col3.metric("Products Affected", optimizer.summary["products_affected"])
            col4.metric("High Priority", optimizer.summary["high_priority"])
            
            # Recommendations table
            st.dataframe(recommendations, use_container_width=True)
            
            # Download
            csv = recommendations.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Recommendations", csv,
                             "warehouse_transfers.csv", "text/csv")
            
            # Visualization
            st.markdown("### 📊 Transfer Flow")
            fig = px.bar(
                recommendations,
                x="destination_warehouse",
                y="transfer_quantity",
                color="source_warehouse",
                title="Transfer Quantity by Destination Warehouse",
                labels={"destination_warehouse": "Destination", 
                       "transfer_quantity": "Transfer Quantity",
                       "source_warehouse": "Source"}
            )
            fig.update_layout(template="plotly_white", height=400)
            render_plotly(fig, use_container_width=True)
            
        else:
            st.info("No transfer recommendations generated. Inventory is balanced across warehouses.")
    except Exception as e:
        st.error(f"Recommendation generation failed: {e}")

st.markdown("---")
st.markdown("### 📝 Methodology")
st.markdown("""
**Optimization Approach:**
1. **Analyze** current inventory and demand per product-warehouse pair
2. **Calculate** target inventory = demand × (safety_days + lead_time)
3. **Identify** surplus warehouses (inventory > target × 1.2)
4. **Identify** shortage warehouses (inventory < target)
5. **Recommend** transfers from surplus to shortage warehouses

**Constraints:**
- Minimum transfer quantity (configurable)
- Safety stock maintenance at source
- Priority based on destination inventory status

*This is a transparent, rule-based approach. Results are simulated recommendations, not optimized solutions.*
""")