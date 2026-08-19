from __future__ import annotations

import numpy as np


def compute_regression_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / MAPE(+1, 0-수요 스무딩). train.py의 Trainer.compute_metrics와
    test.py의 평가 루프가 이 함수 하나를 공유한다."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    diff = labels - preds

    return {
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'mae': float(np.mean(np.abs(diff))),
        'mape_plus1': float(np.mean(np.abs(diff) / (np.abs(labels) + 1)) * 100),
    }
