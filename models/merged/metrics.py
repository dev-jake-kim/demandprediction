from __future__ import annotations

import numpy as np

from ..metrics import compute_regression_metrics


def compute_merged_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """저장소 공용 RMSE/MAE/MAPE(+1)에 merged_model의 MAPE(0제외)를 더한 것.

    원본 ``merged_model/train.py``의 ``run_epoch()``가 내던 지표 집합과 같아야
    ``docs/MERGED_ABLATION_RESULTS.md``의 기존 컬럼과 계속 비교할 수 있다.
    MAPE(0제외)는 실제 수요가 0인 셀을 분자/분모 양쪽에서 빼고 계산한 것이다.
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
