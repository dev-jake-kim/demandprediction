"""Final neural prediction head."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class OutputHead(nn.Module):
    """``h_attn [B,N,F]`` -> 노드별 수요 예측 ``[B,N]`` (Linear, 선택적으로 Softplus)."""

    def __init__(self, fusion_dim: int, use_softplus: bool = True) -> None:
        super().__init__()
        self.neural_head = nn.Linear(fusion_dim, 1)
        self.use_softplus = use_softplus

    def forward(self, h_attn: Tensor) -> Tensor:
        raw = self.neural_head(h_attn).squeeze(-1)
        return F.softplus(raw) if self.use_softplus else raw


__all__ = ["OutputHead"]
