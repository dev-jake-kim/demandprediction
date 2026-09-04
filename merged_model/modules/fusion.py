"""Final neural/retrieval prediction gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuralRetrievalGate(nn.Module):
    """another_model-style raw-scale neural/retrieval output gate."""

    def __init__(self, fusion_dim: int, use_retrieval: bool = True) -> None:
        super().__init__()
        self.use_retrieval = use_retrieval
        self.neural_head = nn.Linear(fusion_dim, 1)
        # 검색기를 끄면 게이트가 섞을 대상이 없으므로 lambda_layer 자체를 만들지 않는다
        # (만들어두면 state_dict에 죽은 파라미터가 남아 체크포인트 해석이 헷갈린다).
        self.lambda_layer = nn.Linear(fusion_dim + 1, 1) if use_retrieval else None

    def forward(self, h_attn: Tensor, ir_out: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        neural_pred = F.softplus(self.neural_head(h_attn)).squeeze(-1)
        if not self.use_retrieval:
            # lambda=1로 고정한 것과 같다 — 예측이 전적으로 뉴럴 브랜치에서 나온다.
            return neural_pred, torch.ones_like(neural_pred), neural_pred
        if ir_out is None:
            raise ValueError('use_retrieval=True인데 ir_out이 None임')
        lambda_weight = torch.sigmoid(
            self.lambda_layer(torch.cat([h_attn, ir_out.unsqueeze(-1)], dim=-1))
        ).squeeze(-1)
        prediction = lambda_weight * neural_pred + (1.0 - lambda_weight) * ir_out
        return neural_pred, lambda_weight, prediction


__all__ = ["NeuralRetrievalGate"]
