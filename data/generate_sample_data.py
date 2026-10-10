"""
SupplyGuard - Synthetic Sample Dataset Generator
Generates a realistic synthetic supply chain dataset for demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_sample_dataset(
    n_days: int = 365,
    n_products: int = 10,
    n_warehouses: int = 4,
    n_suppliers: int = 5,
    output_path: str = "data/sample_supply_chain.csv"
) -> pd.DataFrame:
    """
    Generate a synthetic supply chain dataset with realistic relationships.
    
    This is a SYNTHETIC DEMONSTRATION DATASET - not real-world data.
    """
    products = [f"PROD_{i:03d}" for i in range(1, n_products + 1)]
    categories = ["Electronics", "Apparel", "Home", "Food", "Sports"]
    warehouses = [f"WH_{chr(65+i)}" for i in range(n_warehouses)]
    suppliers = [f"SUP_{i:03d}" for i in range(1, n_suppliers + 1)]
    
    # Assign categories to products
    product_category = {p: random.choice(categories) for p in products}
    
    # Base demand per product (varies by product)
    base_demand = {p: np.random.uniform(50, 200) for p in products}
    
    # Base price per product
    base_price = {p: np.random.uniform(10, 100) for p in products}
    
    # Supplier lead times
    supplier_lead_time = {s: np.random.uniform(3, 15) for s in suppliers}
    
    start_date = datetime(2023, 1, 1)
    records = []
    
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        
        for product in products:
            for warehouse in warehouses:
                supplier = random.choice(suppliers)
                
                # Seasonal component
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
                # Weekly component (weekends lower)
                weekly = 0.8 if current_date.weekday() >= 5 else 1.0
                # Trend component
                trend = 1 + 0.0005 * day
                
                # Promotion effect
                promotion = 1 if random.random() < 0.1 else 0
                promo_effect = 1.5 if promotion else 1.0
                
                # Holiday effect
                holiday = 1 if (current_date.month == 12 and current_date.day in [25, 26]) or \
                              (current_date.month == 1 and current_date.day == 1) else 0
                holiday_effect = 1.8 if holiday else 1.0
                
                # Random noise
                noise = np.random.normal(1, 0.15)
                
                # Compute demand
                demand = base_demand[product] * seasonal * weekly * trend * \
                         promo_effect * holiday_effect * noise
                demand = max(0, int(demand))
                
                # Price with slight variation
                price = base_price[product] * np.random.uniform(0.95, 1.05)
                if promotion:
                    price *= 0.85
                
                # Competitor price
                competitor_price = price * np.random.uniform(0.9, 1.1)
                
                # Lead time
                lead_time = supplier_lead_time[supplier] * np.random.uniform(0.8, 1.2)
                
                # Inventory (simplified: based on demand and lead time)
                inventory = int(np.random.uniform(0.5, 2.0) * demand * lead_time)
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": product,
                    "category": product_category[product],
                    "warehouse": warehouse,
                    "supplier_id": supplier,
                    "price": round(price, 2),
                    "promotion": promotion,
                    "holiday": holiday,
                    "inventory": inventory,
                    "lead_time": round(lead_time, 1),
                    "competitor_price": round(competitor_price, 2),
                    "demand": demand
                })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


if __name__ == "__main__":
    generate_sample_dataset()