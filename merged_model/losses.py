"""Training objectives for the unified model.

``CombinedLoss`` is a byte-for-byte port of the shared harness loss in this
repository (``models/losses.py``) so that a ``merged_model`` run and a
``baseline``/``ir-weather``/``ADFormer`` run optimize exactly the same
objective. The original ``merged_model`` objective (raw-scale MAE) stays
available as ``'mae'``.

Every loss here uses ``reduction='none'`` — the model reduces once, so the
per-epoch aggregate can be a true element mean rather than a mean of batch
means (batches differ in size because ``drop_last=False``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CombinedLoss(nn.Module):
    """절대오차 제곱 + gamma * 상대오차 제곱. y_true가 0에 가까울수록 상대오차 항이
    분모를 eps로 안정화한다."""

    def __init__(self, gamma: float = 1.0, eps: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_true - y_pred
        term1 = diff ** 2

        relative_diff = diff / (y_true + self.eps)
        term2 = relative_diff ** 2

        return term1 + self.gamma * term2


def build_loss(loss_type: str, *, gamma: float = 1.0, eps: float = 0.5) -> nn.Module:
    """'combined'(저장소 공용 CombinedLoss) 또는 'mae'(raw 스케일 L1)를 만든다.

    'mae'는 merged_model이 원래 쓰던 목적함수(``F.l1_loss``)와 동일하고, 'combined'는
    이 저장소의 다른 모든 모델과 손실을 맞추기 위한 것이다. 'mae'일 때 gamma/eps는
    쓰이지 않는다.
    """

    if loss_type == "combined":
        return CombinedLoss(gamma=gamma, eps=eps)
    if loss_type == "mae":
        return nn.L1Loss(reduction="none")
    raise ValueError(f"알 수 없는 loss_type: {loss_type!r} (가능: 'combined', 'mae')")


__all__ = ["CombinedLoss", "build_loss"]
