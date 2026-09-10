"""
Optuna Bayesian hyperparameter optimization script for LightGBM, XGBoost, and CatBoost
using 5-fold Time-Series Cross-Validation and Huber loss objectives.
"""

import os
import sys
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.features import FEATURE_COLUMNS, TARGET, TIMESTAMP, add_derived_features

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "dam_price_prediction_model_ready.csv")
    df = pd.read_csv(data_path)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)

    if "mcp_same_block_std7d" not in df.columns:
        df = add_derived_features(df)

    df_clean = df.dropna(subset=FEATURE_COLUMNS + [TARGET]).reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.80)
    train_df = df_clean.iloc[:split_idx]

    return train_df[FEATURE_COLUMNS], train_df[TARGET].values


def objective_lgb(trial, X, y):
    params = {
        "objective": "huber",
        "huber_alpha": trial.suggest_float("huber_alpha", 0.5, 3.0),
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, preds))

    return np.mean(maes)


def objective_xgb(trial, X, y):
    params = {
        "objective": "reg:pseudohubererror",
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, preds))

    return np.mean(maes)


def objective_cb(trial, X, y):
    params = {
        "loss_function": "Huber:delta=1.5",
        "iterations": trial.suggest_int("iterations", 150, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_seed": 42,
        "verbose": 0,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, preds))

    return np.mean(maes)


def main():
    X, y = load_data()
    best_params = {}

    print("Running Optuna tuning for LightGBM...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda t: objective_lgb(t, X, y), n_trials=30)
    best_params["LightGBM"] = study_lgb.best_params
    print(f"LightGBM Best CV MAE: {study_lgb.best_value:.2f}")

    print("Running Optuna tuning for XGBoost...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda t: objective_xgb(t, X, y), n_trials=30)
    best_params["XGBoost"] = study_xgb.best_params
    print(f"XGBoost Best CV MAE: {study_xgb.best_value:.2f}")

    print("Running Optuna tuning for CatBoost...")
    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(lambda t: objective_cb(t, X, y), n_trials=30)
    best_params["CatBoost"] = study_cb.best_params
    print(f"CatBoost Best CV MAE: {study_cb.best_value:.2f}")

    out_path = os.path.join(os.path.dirname(__file__), "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Saved tuned hyperparameters to {out_path}")


if __name__ == "__main__":
    main()
