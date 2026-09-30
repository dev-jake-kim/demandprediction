from __future__ import annotations

import numpy as np


def compute_regression_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and MAPE(+1) in percent units."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    diff = labels - preds

    return {
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'mae': float(np.mean(np.abs(diff))),
        'mape_plus1': float(np.mean(np.abs(diff) / (np.abs(labels) + 1)) * 100),
    }
