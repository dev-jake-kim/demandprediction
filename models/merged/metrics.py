from __future__ import annotations

import numpy as np

from ..metrics import compute_regression_metrics


def compute_merged_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE(+1), and MAPE excluding zero-demand cells.

    ``mape_excl_zero`` is NaN when all labels are zero.
    """

    metrics = compute_regression_metrics(preds, labels)
    flat_preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    flat_labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    nonzero = flat_labels != 0
    metrics['mape_excl_zero'] = (
        float(
            np.mean(
                np.abs(flat_labels[nonzero] - flat_preds[nonzero]) / np.abs(flat_labels[nonzero])
            )
            * 100
        )
        if bool(nonzero.any())
        else float('nan')
    )
    return metrics


__all__ = ['compute_merged_metrics']
