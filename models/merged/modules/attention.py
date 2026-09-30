"""Attention that selects among daily, weekly, and neural representations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BranchAttention(nn.Module):
    """Node-wise attention over branch tokens ``[daily, weekly, h_neural]``."""

    def __init__(
        self,
        history_hidden: int,
        periodic_hidden: int,
        fusion_dim: int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.neural_projection = nn.Linear(history_hidden, fusion_dim)
        self.daily_projection = nn.Linear(periodic_hidden, fusion_dim)
        self.weekly_projection = nn.Linear(periodic_hidden, fusion_dim)
        self.query_projection = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.key_projection = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.value_projection = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        h_neural: Tensor,
        h_daily: Tensor,
        h_weekly: Tensor,
        daily_valid: Tensor,
        weekly_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        z_neural = self.neural_projection(h_neural)
        z_daily = self.daily_projection(h_daily)
        z_weekly = self.weekly_projection(h_weekly)
        candidates = torch.stack([z_daily, z_weekly, z_neural], dim=2)
        query = self.query_projection(z_neural).unsqueeze(2)
        keys = self.key_projection(candidates)
        values = self.value_projection(candidates)
        scores = (query * keys).sum(dim=-1) / math.sqrt(candidates.shape[-1])

        batch, nodes = h_neural.shape[:2]
        candidate_valid = torch.stack(
            [
                daily_valid[:, None].expand(batch, nodes),
                weekly_valid[:, None].expand(batch, nodes),
                torch.ones(batch, nodes, dtype=torch.bool, device=h_neural.device),
            ],
            dim=-1,
        )
        if self.use_attention:
            scores = scores.masked_fill(~candidate_valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
        else:
            uniform = candidate_valid.to(scores.dtype)
            weights = uniform / uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
        fused = (weights.unsqueeze(-1) * values).sum(dim=2)
        return self.norm(z_neural + self.output_projection(fused)), weights


__all__ = ["BranchAttention"]
