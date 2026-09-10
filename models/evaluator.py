"""
Evaluation module computing standard regression metrics and peak block metrics
for Day-Ahead Market (DAM) electricity price forecasting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE)."""
    sum_true = np.sum(np.abs(y_true))
    if sum_true == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_true * 100.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray | None = None
) -> dict[str, float]:
    """
    Computes complete evaluation metrics suite:
    - MAE (INR / MWh)
    - RMSE (INR / MWh)
    - WAPE (%)
    - R2 Score
    - Peak-Hour MAE (INR / MWh) [Morning 7-10 AM & Evening 6-10 PM]
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    wape = compute_wape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    peak_mae = mae
    if hours is not None:
        peak_mask = np.isin(hours, [7, 8, 9, 10, 18, 19, 20, 21, 22])
        if np.any(peak_mask):
            peak_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]))

    return {
        "MAE (INR/MWh)": round(mae, 2),
        "RMSE (INR/MWh)": round(rmse, 2),
        "WAPE (%)": round(wape, 2),
        "R2 Score": round(r2, 4),
        "Peak-Hour MAE": round(peak_mae, 2),
    }
