# QTFT_Step4_Evaluate.py
# Unified evaluation metrics for multi-horizon forecasting

import math
import torch


def evaluate_multi_horizon(y_true, y_pred, horizons=(1, 4, 8)):
    """
    y_true: Tensor [N, H]
    y_pred: Tensor [N, H]
    """
    assert y_true.shape == y_pred.shape

    metrics = {}

    for h in horizons:
        idx = h - 1
        if idx >= y_true.shape[1]:
            metrics[h] = {"RMSE": None, "MAE": None}
            continue

        t = y_true[:, idx]
        p = y_pred[:, idx]

        mse = torch.mean((p - t) ** 2).item()
        mae = torch.mean(torch.abs(p - t)).item()
        rmse = math.sqrt(mse)

        metrics[h] = {
            "RMSE": rmse,
            "MAE": mae,
        }

    return metrics
