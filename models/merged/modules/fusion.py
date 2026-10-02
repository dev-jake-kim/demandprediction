"""Final neural/retrieval prediction gate and local-representation retrieval fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuralRetrievalGate(nn.Module):
    """Combine neural and retrieval predictions with an optional gate."""

    def __init__(self, fusion_dim: int, use_softplus: bool = True) -> None:
        super().__init__()
        self.neural_head = nn.Linear(fusion_dim, 1)
        self.lambda_layer = nn.Linear(fusion_dim + 1, 1)
        self.use_softplus = use_softplus

    def forward(
        self, h_attn: Tensor, ir_out: Tensor | None, *, bypass_gate: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.neural_head(h_attn).squeeze(-1)
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


class RetrievalLocalFusion(nn.Module):
    """``retrieval_injection='local_concat'``: 검색 결과를 local 표현에 섞는다.

    ``h_local = Linear(2H → H)([h_neural ⊕ Linear(1 → H)(log1p(ir_out))])``, ``H = history_hidden``.
    ``h_local``이 BranchAttention에서 ``h_neural``을 대신한다.
    """

    def __init__(self, history_hidden: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, history_hidden)
        self.fuse = nn.Linear(2 * history_hidden, history_hidden)

    def forward(self, h_neural: Tensor, ir_out: Tensor) -> Tensor:
        r_emb = self.value_projection(torch.log1p(ir_out.clamp_min(0.0)).unsqueeze(-1).to(h_neural.dtype))
        return self.fuse(torch.cat([h_neural, r_emb], dim=-1))


__all__ = ["NeuralRetrievalGate", "RetrievalLocalFusion"]
