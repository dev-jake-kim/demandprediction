"""Final neural/retrieval prediction gate."""

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


__all__ = ["NeuralRetrievalGate"]
