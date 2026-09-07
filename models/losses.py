from __future__ import annotations

import math

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """절대오차 제곱 + gamma * 상대오차 제곱. y_true가 0에 가까울수록 상대오차 항이
    분모를 eps로 안정화한다."""

    def __init__(self, gamma: float = 1.0, eps: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        term1 = diff ** 2

        relative_diff = diff / (y_true + self.eps)
        term2 = relative_diff ** 2

        loss = term1 + self.gamma * term2

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class RmseMapeLoss(nn.Module):
    """``rmse_weight * RMSE + MAPE(+1)``.

    MAPE는 models/metrics.py의 보고 지표와 똑같이 분모 ``abs(y)+1`` 및 퍼센트 단위를 쓴다.
    rmse_weight=10이면 두 항이 (RMSE 0.75 -> 7.5) vs (MAPE 15) 정도로 비슷한 크기가 된다.

    학습 중 RMSE는 미니배치 단위 surrogate라 batch size에 영향을 받는다. validation의 best
    checkpoint 선택은 train.py가 전체 validation 예측으로 다시 계산한 동일 목적함수를 사용한다.
    """

    def __init__(self, rmse_weight: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(rmse_weight) or rmse_weight < 0:
            raise ValueError(f'rmse_weight는 0 이상이어야 함: got {rmse_weight}')
        self.rmse_weight = rmse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        # sqrt(mean(diff**2))와 값은 같지만, vector_norm은 diff가 전부 0일 때도
        # 0 gradient를 돌려준다. sqrt의 0 지점에서 생기는 0/0 gradient를 피하기 위함이다.
        rmse = torch.linalg.vector_norm(diff) / math.sqrt(diff.numel())
        mape = (diff.abs() / (y_true.abs() + 1.0)).mean() * 100.0
        return self.rmse_weight * rmse + mape
