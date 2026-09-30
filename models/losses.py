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
    """Scalar ``rmse_weight * RMSE + MAPE(+1)`` objective.

    MAPE uses denominator ``abs(y_true) + 1`` and percent units. The RMSE spans all
    input elements, so this loss always returns a scalar.
    """

    def __init__(self, rmse_weight: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(rmse_weight) or rmse_weight < 0:
            raise ValueError(f'rmse_weight는 0 이상이어야 함: got {rmse_weight}')
        self.rmse_weight = rmse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        # vector_norm avoids undefined 0/0 gradients at zero error.
        rmse = torch.linalg.vector_norm(diff) / math.sqrt(diff.numel())
        mape = (diff.abs() / (y_true.abs() + 1.0)).mean() * 100.0
        return self.rmse_weight * rmse + mape


class DemandSplitLoss(nn.Module):
    """Elementwise loss that uses MAPE(+1) for low-demand cells and weighted squared error
    above the threshold.

    ``y_true <= threshold`` uses ``abs(error) / (abs(y_true) + 1)``; higher values use
    ``high_weight * error**2``.
    """

    def __init__(self, threshold: float = 1.0, high_weight: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f'threshold는 0 이상이어야 함: got {threshold}')
        if not math.isfinite(high_weight) or high_weight < 0:
            raise ValueError(f'high_weight는 0 이상이어야 함: got {high_weight}')
        self.threshold = threshold
        self.high_weight = high_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        relative = diff.abs() / (y_true.abs() + 1.0)
        squared = diff ** 2
        is_low = y_true <= self.threshold
        return torch.where(is_low, relative, self.high_weight * squared)
