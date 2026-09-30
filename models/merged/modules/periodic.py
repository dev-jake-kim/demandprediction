"""Daily and weekly lag encoders with explicit invalid-lag handling."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PeriodicLSTMEncoder(nn.Module):
    """Encode valid daily/weekly lag values in chronological order."""

    def __init__(self, hidden_size: int, extra_dim: int = 0) -> None:
        super().__init__()
        if extra_dim < 0:
            raise ValueError("extra_dim must be non-negative")
        self.extra_dim = extra_dim
        self.lstm = nn.LSTM(1 + extra_dim, hidden_size, batch_first=True)

    def forward(
        self, values: Tensor, invalid_mask: Tensor, extra: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if values.ndim != 4 or values.shape[-1] != 1:
            raise ValueError(f"Periodic values must be [B,L,N,1], got {tuple(values.shape)}")
        if invalid_mask.ndim != 2 or invalid_mask.shape[:2] != values.shape[:2]:
            raise ValueError("Periodic mask must be [B,L] matching the first two value dimensions")

        batch, length, nodes, _ = values.shape
        valid_mask = ~invalid_mask.bool()
        lengths = valid_mask.sum(dim=1)
        sequence = torch.log1p(torch.clamp(values, min=0.0))

        if self.extra_dim > 0:
            if extra is None:
                raise ValueError(f"extra_dim={self.extra_dim}인데 extra가 None임")
            if extra.shape != (batch, length, self.extra_dim):
                raise ValueError(
                    f"extra must be [B,L,{self.extra_dim}], got {tuple(extra.shape)}"
                )
            sequence = torch.cat(
                [sequence, extra[:, :, None, :].expand(batch, length, nodes, self.extra_dim)],
                dim=-1,
            )
        elif extra is not None:
            raise ValueError("extra_dim=0인데 extra가 주어짐")

        # 유효 lag를 원래 순서대로 앞에 모은다. lag 유효성은 샘플 단위라 노드 축에 공유된다.
        order = valid_mask.to(torch.long).argsort(dim=1, descending=True, stable=True)
        feature_dim = sequence.shape[-1]
        compact = sequence.gather(1, order[:, :, None, None].expand(-1, -1, nodes, feature_dim))
        compact = compact.permute(0, 2, 1, 3).reshape(batch * nodes, length, feature_dim)
        # 전체 길이로 돌리고 각 행의 마지막 유효 시점 출력을 읽는다(packed LSTM의 마지막 hidden과
        # 같다). host 동기화가 필요한 bool index·packing을 쓰지 않는다.
        steps, _ = self.lstm(compact)
        last = (lengths.clamp_min(1) - 1).repeat_interleave(nodes)
        index = last[:, None, None].expand(-1, 1, steps.shape[-1])
        hidden = steps.gather(1, index).squeeze(1)
        row_valid = (lengths > 0).repeat_interleave(nodes)
        output = hidden * row_valid[:, None].to(hidden.dtype)
        return output.reshape(batch, nodes, -1), lengths > 0


__all__ = ["PeriodicLSTMEncoder"]
