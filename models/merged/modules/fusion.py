"""Final neural/retrieval prediction gate."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuralRetrievalGate(nn.Module):
    """Combine neural and retrieval predictions with an optional gate."""

    def __init__(
        self, fusion_dim: int, use_softplus: bool = True, node_adaptive_indices: Sequence[int] | None = None
    ) -> None:
        super().__init__()
        self.neural_head = nn.Linear(fusion_dim, 1)
        self.lambda_layer = nn.Linear(fusion_dim + 1, 1)
        self.use_softplus = use_softplus
        # 선택 노드별 head ΔW(docs/SPEC.md §6.7 node_adaptive_scope=head): raw[n] += h·Δw[n] + Δb[n]. 0으로 생성.
        self._node_index_list = None if node_adaptive_indices is None else [int(v) for v in node_adaptive_indices]
        self._cached_node_index: Tensor | None = None
        if self._node_index_list is not None:
            self.node_delta_head_weight = nn.Parameter(torch.zeros(len(self._node_index_list), fusion_dim))
            self.node_delta_head_bias = nn.Parameter(torch.zeros(len(self._node_index_list)))

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        if self._node_index_list is None:
            return ()
        return (self.node_delta_head_weight, self.node_delta_head_bias)

    def forward(
        self, h_attn: Tensor, ir_out: Tensor | None, *, bypass_gate: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.neural_head(h_attn).squeeze(-1)
        if self._node_index_list is not None:
            index = self._cached_node_index
            if index is None or index.device != h_attn.device:
                index = torch.as_tensor(self._node_index_list, dtype=torch.long, device=h_attn.device)
                self._cached_node_index = index
            delta = torch.einsum("baf,af->ba", h_attn[:, index], self.node_delta_head_weight) + self.node_delta_head_bias
            raw = raw.index_add(1, index, delta.to(raw.dtype))
        neural_pred = F.softplus(raw) if self.use_softplus else raw

        if bypass_gate:
            lambda_weight = torch.ones_like(neural_pred)
            return neural_pred, lambda_weight, neural_pred

        assert ir_out is not None, "bypass_gate=False면 ir_out이 필요함"
        lambda_weight = torch.sigmoid(
            self.lambda_layer(torch.cat([h_attn, ir_out.unsqueeze(-1)], dim=-1))
        ).squeeze(-1)
        prediction = lambda_weight * neural_pred + (1.0 - lambda_weight) * ir_out
        return neural_pred, lambda_weight, prediction


__all__ = ["NeuralRetrievalGate"]
