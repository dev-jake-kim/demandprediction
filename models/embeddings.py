from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScalarEmbedding(nn.Module):
    """스칼라 값(demand) -> d_model 벡터. 다른 임베딩 방식으로 바꿀 땐 forward만 구현해서 교체."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FourierScalarEmbedding(ScalarEmbedding):
    """γ(x) = [cos(2π b_1 x), sin(2π b_1 x), ..., cos(2π b_m x), sin(2π b_m x)], 2m = d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model은 짝수여야 함 (2 * n_freqs): got {d_model}")
        n_freqs = d_model // 2
        self.freqs = nn.Parameter(torch.randn(n_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * x.unsqueeze(-1) * self.freqs  # (..., n_freqs)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., d_model)
