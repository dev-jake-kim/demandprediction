from __future__ import annotations

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
