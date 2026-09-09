"""Scalar-to-token embeddings used by the local history branch."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierScalarEmbedding(nn.Module):
    """Learnable Fourier features followed by a projection."""

    def __init__(self, d_model: int, num_bands: int = 8) -> None:
        super().__init__()
        self.log_frequencies = nn.Parameter(torch.linspace(-2.0, 2.0, num_bands))
        self.projection = nn.Linear(1 + 2 * num_bands, d_model)

    def forward(self, value: Tensor) -> Tensor:
        value = value.unsqueeze(-1) if value.ndim == 0 or value.shape[-1] != 1 else value
        frequencies = self.log_frequencies.exp().view(*([1] * (value.ndim - 1)), -1)
        phase = value * frequencies * (2.0 * math.pi)
        features = torch.cat([value, phase.sin(), phase.cos()], dim=-1)
        return self.projection(features)


__all__ = ["FourierScalarEmbedding"]
