# SupplyGuard

## An Explainable AI Framework for Risk-Aware and Sustainable Supply Chain Management

**Version:** 1.0.0

---

## Abstract

SupplyGuard is a research-oriented B.Tech final-year major project that integrates demand forecasting, stock-out early warning, supplier risk prediction, demand anomaly detection, multi-warehouse redistribution, explainable AI (SHAP), sustainability indicators, and what-if simulation into a unified decision-support system for supply chain management.

The framework provides **predictive intelligence** (what is likely to happen), **prescriptive intelligence** (what should be considered), and **explainable intelligence** (why the model produced a prediction).

---

## Problem Statement

Supply chains experience uncertain demand, stock-outs, excess inventory, supplier risks, warehouse imbalance, demand spikes/drops, long lead times, price fluctuations, and operational inefficiencies. Existing solutions often address these challenges in isolation, lacking integration and explainability.

---

## Objectives

1. **Predictive Stock-Out Early Warning** — Predict products at risk of stock-out
2. **Supplier Risk Prediction** — Identify high-risk suppliers
3. **Multi-Warehouse Redistribution** — Optimize inventory across warehouses
4. **SHAP Explainable AI** — Explain every major prediction
5. **Demand Anomaly Detection** — Detect unusual demand behavior

---

## Features

| Module | Description |
|--------|-------------|
| Demand Forecasting | Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM |
| Stock-Out Prediction | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| Supplier Risk | Operational feature-based classification |
| Anomaly Detection | Isolation Forest, Z-score, IQR |
| Warehouse Optimization | Surplus/shortage analysis with transfer recommendations |
| Explainable AI | SHAP global and local explanations |
| Sustainability | Relative sustainability indicators |
| What-If Simulation | Scenario analysis |
| Research Experiments | Reproducible experiment logging |

## Research Protocol and Limitations

Demand forecasting keeps all rows from the same date in one chronological train, validation, or test split. It creates calendar and shifted historical demand features, tunes LightGBM over four parameter settings with three expanding-window folds inside training data, selects models using validation RMSE, refits candidates on training plus validation data, and reports performance on the later held-out test period. Same-period inventory is excluded from demand predictors because the bundled synthetic generator derives inventory from demand. Regression evaluation includes RMSE, MAE, MAPE, WAPE, sMAPE, and MASE. These changes improve evaluation discipline, but a single holdout on synthetic data cannot establish real-world accuracy or be compared directly with the reference paper's UCI retail results.

CSV import normalizes common aliases such as `Units_Sold` to `demand`, `Inventory_Level` to `inventory`, `SKU_ID` to `product_id`, `Warehouse_ID` to `warehouse`, and `Supplier_Lead_Time_Days` to `lead_time`. A supplied `Demand_Forecast` column is reported as a benchmark, not used as a model input unless its timing and provenance are verified. If a supplied stock-out flag contains only one class, the stock-out module uses an explicitly labeled inventory-demand rule; supplier-risk labels are rule-derived in this research demo and are not ground-truth outcomes.

The Research Experiments page uses a chronological holdout when the processed dataset contains a valid datetime column. Entire dates are assigned to one side of the split, and later dates form the test set. Without usable dates, it uses a seeded random split, stratified for classification when class counts allow. Each run compares the selected estimator with a mean (regression) or most-frequent-class (classification) baseline.

Experiment exports include the target, selected features, split method, seed, test size, model settings, software versions, metrics, baseline metrics, and a SHA-256 fingerprint of the processed input table. Keep the exported JSON with any reported results; a fingerprint identifies the exact input but does not archive or recover the dataset.

The bundled supply-chain data is synthetic and is suitable for demonstration and software tests only. Results on it do not establish operational effectiveness or generalization to real supply chains. The experiment page receives an already-preprocessed table; current missing-value imputation is performed before splitting, so imputation statistics can include test-period information. For publication-grade evaluation, move imputation and any learned preprocessing into a pipeline fitted separately on each training split. Also evaluate on documented real-world data, compare against domain-appropriate methods, report uncertainty across repeated periods or sites, audit feature/label timing, and document ethics, data provenance, and limitations. The current page is an experiment runner, not a complete validation study; ablation testing and repeated out-of-time evaluation are not implemented.

---

## System Architecture
