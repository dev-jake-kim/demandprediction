"""Final neural/retrieval prediction gate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuralRetrievalGate(nn.Module):
    """another_model-style raw-scale neural/retrieval output gate."""

    def __init__(self, fusion_dim: int, use_softplus: bool = True) -> None:
        super().__init__()
        self.neural_head = nn.Linear(fusion_dim, 1)
        self.lambda_layer = nn.Linear(fusion_dim + 1, 1)
        # False면 neural_head의 raw 선형 출력을 그대로 쓴다 — softplus(x)=log(1+exp(x))로
        # 양수를 강제하던 마지막 보정을 없애는 ablation. 음수 예측이 나올 수 있다.
        self.use_softplus = use_softplus

    def forward(
        self, h_attn: Tensor, ir_out: Tensor | None, *, bypass_gate: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.neural_head(h_attn).squeeze(-1)
        neural_pred = F.softplus(raw) if self.use_softplus else raw

        if bypass_gate:
            # no-ir ablation 전용 경로. 예전에는 ir_out=0을 게이트에 그대로 흘려보내
            # prediction = lambda*neural_pred로 만들었는데, zero-inflated 수요에서
            # "전부 0 예측"이 손실의 국소 최적이라 lambda->0으로 미끄러지면 neural_pred
            # 쪽 gradient도 lambda에 곱해져 함께 사라져 다시 못 빠져나오는 죽음의 함정이었다
            # (실측: porto 4시드 중 3개가 RMSE 3.7975로 소수점까지 동일하게 붕괴).
            # lambda_layer 자체를 아예 거치지 않고 neural_pred를 그대로 예측값으로 쓴다 —
            # 게이트가 곱해지는 경로가 없으니 이 함정이 구조적으로 발생할 수 없다.
            lambda_weight = torch.ones_like(neural_pred)
            return neural_pred, lambda_weight, neural_pred

        assert ir_out is not None, "bypass_gate=False면 ir_out이 필요함"
        lambda_weight = torch.sigmoid(
            self.lambda_layer(torch.cat([h_attn, ir_out.unsqueeze(-1)], dim=-1))
        ).squeeze(-1)
        prediction = lambda_weight * neural_pred + (1.0 - lambda_weight) * ir_out
        return neural_pred, lambda_weight, prediction


__all__ = ["NeuralRetrievalGate"]
