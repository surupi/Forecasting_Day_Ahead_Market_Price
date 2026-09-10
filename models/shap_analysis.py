"""
SHAP (SHapley Additive exPlanations) analysis module for identifying key price-driving features.
"""

import os
import sys
import numpy as np
import pandas as pd
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.features import FEATURE_COLUMNS, TARGET, TIMESTAMP, add_derived_features
import lightgbm as lgb


def run_shap_analysis():
    data_path = os.path.join(os.path.dirname(__file__), "..", "dam_price_prediction_model_ready.csv")
    df = pd.read_csv(data_path)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)

    if "mcp_same_block_std7d" not in df.columns:
        df = add_derived_features(df)

    df_clean = df.dropna(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.80)
    X_train = df_clean.iloc[:split_idx][FEATURE_COLUMNS]
    y_train = df_clean.iloc[:split_idx][TARGET].values
    X_test = df_clean.iloc[split_idx:][FEATURE_COLUMNS]

    # Train LightGBM for SHAP
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Compute mean absolute SHAP value for feature importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Mean Abs SHAP Value (INR/MWh Impact)": mean_abs_shap
    }).sort_values(by="Mean Abs SHAP Value (INR/MWh Impact)", ascending=False).reset_index(drop=True)

    shap_importance["Mean Abs SHAP Value (INR/MWh Impact)"] = shap_importance["Mean Abs SHAP Value (INR/MWh Impact)"].round(2)

    return shap_importance


if __name__ == "__main__":
    df_shap = run_shap_analysis()
    print("Top 10 Key Price Drivers (SHAP Value Impact):")
    print(df_shap.head(10).to_string(index=False))
