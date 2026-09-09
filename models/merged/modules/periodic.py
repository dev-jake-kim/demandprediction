"""Daily and weekly lag encoders with explicit invalid-lag handling."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


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
        row_lengths = lengths.repeat_interleave(nodes)
        row_valid = row_lengths > 0
        sequence = torch.log1p(torch.clamp(values, min=0.0))
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * nodes, length, 1)

        # 날씨/캘린더는 노드에 무관하므로 [B,L,E]를 노드 축으로 브로드캐스트해 붙인다.
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError(f"extra_dim={self.extra_dim}인데 extra가 None임")
            if extra.shape != (batch, length, self.extra_dim):
                raise ValueError(
                    f"extra must be [B,L,{self.extra_dim}], got {tuple(extra.shape)}"
                )
            expanded_extra = (
                extra[:, None, :, :]
                .expand(batch, nodes, length, self.extra_dim)
                .reshape(batch * nodes, length, self.extra_dim)
            )
            sequence = torch.cat([sequence, expanded_extra], dim=-1)
        elif extra is not None:
            raise ValueError("extra_dim=0인데 extra가 주어짐")

        feature_dim = sequence.shape[-1]
        expanded_valid = valid_mask[:, None, :].expand(batch, nodes, length).reshape(batch * nodes, length)
        output = sequence.new_zeros((batch * nodes, self.lstm.hidden_size))

        if row_valid.any():
            valid_sequence = sequence[row_valid]
            valid_positions = expanded_valid[row_valid]
            order = valid_positions.to(dtype=torch.long).argsort(dim=1, descending=True, stable=True)
            # feature 차원을 하드코딩하면 concat한 채널이 조용히 잘려나간다 — 반드시 실제 폭을 쓴다.
            compact = valid_sequence.gather(1, order.unsqueeze(-1).expand(-1, -1, feature_dim))
            compact_lengths = row_lengths[row_valid]
            packed = pack_padded_sequence(
                compact,
                compact_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            output[row_valid] = hidden[-1]
        return output.reshape(batch, nodes, -1), lengths > 0


__all__ = ["PeriodicLSTMEncoder"]
