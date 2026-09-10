"""
Level-2 Meta-Learner Stacking and Out-Of-Fold (OOF) Prediction module.
Trains base estimators using 5-fold Time-Series Cross Validation and fits a Level-2
Meta-Learner (Non-Negative Lasso/Ridge) to dynamically blend base predictions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error


def generate_oof_predictions(models_dict: dict, X_train: pd.DataFrame, y_train: np.ndarray, n_splits: int = 5):
    """
    Generates Out-of-Fold (OOF) predictions for Level-2 Meta-Learner training.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_names = list(models_dict.keys())
    oof_matrix = np.zeros((len(X_train), len(model_names)))

    valid_indices = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        valid_indices.extend(val_idx)
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        for idx, (name, model_fn) in enumerate(models_dict.items()):
            model = model_fn()
            model.fit(X_tr, y_tr)
            oof_matrix[val_idx, idx] = model.predict(X_va)

    # Filter only samples that were part of validation folds
    valid_mask = np.array(valid_indices)
    return oof_matrix[valid_mask], y_train[valid_mask]


def train_stacking_meta_learner(models_dict: dict, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame):
    """
    Trains Level-1 models on full training data, and fits a Level-2 Lasso Meta-Learner.
    Returns:
        meta_preds: Test predictions from stacked ensemble.
        weights: Weights assigned by Meta-Learner to each base model.
    """
    # 1. Generate OOF predictions for Meta-Learner training
    oof_X, oof_y = generate_oof_predictions(models_dict, X_train, y_train)

    # 2. Fit Meta-Learner (Positive coefficients constraint so weights sum to valid blend)
    meta_model = Ridge(alpha=1.0, positive=True)
    meta_model.fit(oof_X, oof_y)

    # 3. Fit base models on full X_train and predict X_test
    test_preds_matrix = np.zeros((len(X_test), len(models_dict)))
    for idx, (name, model_fn) in enumerate(models_dict.items()):
        model = model_fn()
        model.fit(X_train, y_train)
        test_preds_matrix[:, idx] = model.predict(X_test)

    # 4. Final Meta-Learner Predictions
    stacked_preds = meta_model.predict(test_preds_matrix)

    # Normalize weights
    raw_weights = meta_model.coef_
    weight_sum = np.sum(raw_weights)
    weights = raw_weights / weight_sum if weight_sum > 0 else raw_weights

    weights_dict = {name: round(float(w), 4) for name, w in zip(models_dict.keys(), weights)}

    return stacked_preds, weights_dict
