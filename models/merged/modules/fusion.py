"""Node-by-hour softmax fusion of local, daily, and weekly predictions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

GATE_INIT = (0.7, 0.2, 0.1)  # local, daily, weekly


class NodeHourGate(nn.Module):
    """``(노드, 예측 시각)``별 softmax 가중치로 세 예측을 섞는다."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        init = torch.tensor([math.log(value) for value in GATE_INIT])
        self.gate_logit = nn.Parameter(init.expand(num_nodes, 24, len(GATE_INIT)).clone())

    def forward(
        self,
        local_pred: Tensor,
        daily_pred: Tensor,
        weekly_pred: Tensor,
        hour: Tensor,
        daily_valid: Tensor,
        weekly_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """``[B,N]`` 예측 셋, ``[B]`` 시각·유효 -> (``[B,N]`` 예측, ``[B,N,3]`` 가중치)."""

        logits = self.gate_logit[:, hour].transpose(0, 1)  # [B,N,3]
        usable = torch.stack(
            [torch.ones_like(daily_valid), daily_valid, weekly_valid], dim=-1
        ).bool()  # [B,3]
        logits = logits.masked_fill(~usable[:, None, :], float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        stacked = torch.stack([local_pred, daily_pred, weekly_pred], dim=-1)
        return (weights * stacked).sum(dim=-1), weights


__all__ = ["GATE_INIT", "NodeHourGate"]
