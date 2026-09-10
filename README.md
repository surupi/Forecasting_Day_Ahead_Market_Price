# Day-Ahead Market (DAM) Electricity Price Prediction

A machine learning and exploratory analysis project for forecasting electricity **Market Clearing Prices (`mcp_inr_per_mwh`)** in the Day-Ahead Market (DAM) at 15-minute intervals.

---

## 📌 Project Overview

This repository provides end-to-end tools for:
- **Exploratory Data Analysis & Feature Engineering**: Automated scripts profiling market dynamics, temporal features, rolling statistics, demand-supply ratios, and price lags.
- **Model Training & Benchmarking**: Standardized execution of LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Ridge, and Stacking Ensembles.
- **Hyperparameter Optimization**: Automated Optuna optimization for tree-based estimators.
- **Model Interpretability**: Feature importance analysis and SHAP (SHapley Additive exPlanations) values to interpret key market drivers.

---

## 📁 Repository Structure

```
├── dam_price_prediction_eda.ipynb      # Complete EDA & feature exploration notebook
├── dam_price_prediction_model_ready.csv # Preprocessed dataset with engineered features
├── findings.md                          # Comprehensive EDA summary & feature documentation
├── results.md                           # Model evaluation and benchmark summary
├── models/                              # Core machine learning & evaluation module
│   ├── features.py                      # Feature definition lists & categorical mappings
│   ├── evaluator.py                     # Evaluation metrics (RMSE, MAE, R², MAPE)
│   ├── train_and_benchmark.py           # Training pipeline & benchmark runner across models
│   ├── tune_optuna.py                   # Optuna hyperparameter tuning script
│   ├── stacking_meta_learner.py         # Stacking ensemble meta-learner training
│   ├── shap_analysis.py                 # SHAP model interpretability script
│   └── best_params.json                 # Saved tuned parameters from Optuna runs
└── scripts/                             # Modular exploratory pipeline scripts
    ├── stage1_profiling.py              # Data distribution & summary statistics
    ├── stage2_target_timeseries.py      # Time-series target analysis
    ├── stage3_relationships.py          # Bivariate & multivariate correlation analysis
    ├── stage4_outliers.py               # Outlier detection & distribution bounds
    └── stage5_modeling_strategy.py      # Modeling strategy & initial validation checks
```

---

## 📊 Dataset & Feature Engineering

The dataset consists of **7,968 valid records** at 15-minute block intervals (96 blocks per day) across 41 columns:
- **Raw Measurements**: `timestamp`, `block_15min`, `demand_mw`, `solar_generation_mw`, `wind_generation_mw`, `hydro_generation_mw`, `mcp_inr_per_mwh`.
- **Temporal & Cyclic**: `hour_sin`, `hour_cos`, `block_sin`, `block_cos`, `is_weekend`, `peak_block_flag`, `is_solar_window`.
- **Domain Metrics**: `renewable_penetration_ratio`, `supply_cushion_mw`, `residual_demand_mw`, `bid_spread_mw`, `clearing_ratio`.
- **Lags & Rolling Statistics**: `mcp_lag_96` (1-day lag), `mcp_lag_672` (7-day lag), `mcp_same_block_ma7d`, `mcp_same_block_std7d`, `mcp_lag96_rolling_mean_4h`, `mcp_lag96_rolling_std_4h`.

*For detailed column descriptions and market insights, see [eda_findings.md](eda_findings.md).*

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd new
   ```

2. **Install requirements**:
   Make sure Python 3.9+ is installed, then install core dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna shap matplotlib seaborn
   ```

---

## 🚀 Running the Machine Learning Pipeline

### 1. Train and Benchmark Base Models
Run `train_and_benchmark.py` to evaluate multiple algorithms (LightGBM, XGBoost, CatBoost, Random Forest, etc.):
```bash
python -m models.train_and_benchmark
```

### 2. Hyperparameter Tuning with Optuna
Fine-tune LightGBM, XGBoost, and CatBoost hyperparameters using Optuna:
```bash
python -m models.tune_optuna
```

### 3. Train Stacking Meta-Learner
Combine top-performing models using a Ridge meta-learner:
```bash
python -m models.stacking_meta_learner
```

### 4. Perform SHAP Interpretability Analysis
Generate SHAP summary plots and feature importance rankings:
```bash
python -m models.shap_analysis
```

---

## 📈 Key Findings & Benchmark Summary

- **Primary Driver**: 1-day lagged price (`mcp_lag_96`) and 7-day same-block moving average (`mcp_same_block_ma7d`) are the strongest predictors of current clearing price.
- **Reserve Cushion**: Low `supply_cushion_mw` strongly correlates with price spikes during morning and evening peak demand windows.
- *Detailed performance metrics and model comparisons can be viewed in [model_benchmark_performance.md](model_benchmark_performance.md).*
