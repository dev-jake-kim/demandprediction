"""Attention that selects among daily, weekly, and neural representations."""

from __future__ import annotations

import math
from collections.abc import Sequence

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
        node_adaptive_indices: Sequence[int] | None = None,
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
        # 선택 노드별 fusion ΔW(docs/SPEC.md §6.7 node_adaptive_scope=fusion): query += Δq[n] (branch 선호),
        # branch token마다 Δv[n, branch] (노드별 branch 출력 보정). 0으로 생성.
        self._node_index_list = None if node_adaptive_indices is None else [int(v) for v in node_adaptive_indices]
        self._cached_node_index: Tensor | None = None
        if self._node_index_list is not None:
            count = len(self._node_index_list)
            self.node_delta_query = nn.Parameter(torch.zeros(count, fusion_dim))
            self.node_delta_value = nn.Parameter(torch.zeros(count, 3, fusion_dim))

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        if self._node_index_list is None:
            return ()
        return (self.node_delta_query, self.node_delta_value)

    def _node_offset(self, delta: Tensor, nodes: int, device: torch.device) -> Tensor:
        """선택 노드에만 ``delta [A, ...]``를 놓은 ``[N, ...]`` (나머지 0)."""

        index = self._cached_node_index
        if index is None or index.device != device:
            index = torch.as_tensor(self._node_index_list, dtype=torch.long, device=device)
            self._cached_node_index = index
        full = delta.new_zeros((nodes, *delta.shape[1:]))
        return full.index_copy(0, index, delta)

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
        if self._node_index_list is not None:
            nodes = h_neural.shape[1]
            query = query + self._node_offset(self.node_delta_query, nodes, h_neural.device)[None, :, None, :]
            values = values + self._node_offset(self.node_delta_value, nodes, h_neural.device)[None]
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
