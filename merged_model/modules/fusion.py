"""Final neural/retrieval prediction gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuralRetrievalGate(nn.Module):
    """another_model-style raw-scale neural/retrieval output gate."""

    def __init__(self, fusion_dim: int) -> None:
        super().__init__()
        self.neural_head = nn.Linear(fusion_dim, 1)
        self.lambda_layer = nn.Linear(fusion_dim + 1, 1)

    def forward(self, h_attn: Tensor, ir_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # 검색기 ablation은 호출부에서 ir_out을 0으로 바꿔 전달한다 — 게이트 구조는 그대로다.
        # 0을 섞으면 예측이 lambda배로 줄지만, 처음부터 재학습하므로 lambda->1을 배워 보정한다.
        neural_pred = F.softplus(self.neural_head(h_attn)).squeeze(-1)
        lambda_weight = torch.sigmoid(
            self.lambda_layer(torch.cat([h_attn, ir_out.unsqueeze(-1)], dim=-1))
        ).squeeze(-1)
        prediction = lambda_weight * neural_pred + (1.0 - lambda_weight) * ir_out
        return neural_pred, lambda_weight, prediction


__all__ = ["NeuralRetrievalGate"]
