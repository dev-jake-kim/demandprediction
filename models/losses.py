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


def build_loss(config) -> nn.Module:
    """config.loss_type에 따라 손실 모듈을 만든다.

    'combined'(기본)은 이 저장소의 모든 기존 실험이 쓴 CombinedLoss고, 'mae'는 raw 스케일
    L1으로 ADFormer 공식 구현(utils/ADFormer_trainer.py의 F.l1_loss)과 목적함수를 맞추기
    위한 것이다. 'mae'일 때 loss_gamma/loss_eps는 쓰이지 않는다.
    """
    loss_type = getattr(config, 'loss_type', 'combined')
    if loss_type == 'combined':
        return CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)
    if loss_type == 'mae':
        return nn.L1Loss()
    raise ValueError(f"알 수 없는 loss_type: {loss_type!r} (가능: 'combined', 'mae')")
