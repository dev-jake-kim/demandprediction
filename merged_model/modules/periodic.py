"""Daily and weekly lag encoders with explicit invalid-lag handling."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class PeriodicLSTMEncoder(nn.Module):
    """Encode valid daily/weekly lag values in chronological order."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)

    def forward(self, values: Tensor, invalid_mask: Tensor) -> tuple[Tensor, Tensor]:
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
        expanded_valid = valid_mask[:, None, :].expand(batch, nodes, length).reshape(batch * nodes, length)
        output = sequence.new_zeros((batch * nodes, self.lstm.hidden_size))

        if row_valid.any():
            valid_sequence = sequence[row_valid]
            valid_positions = expanded_valid[row_valid]
            order = valid_positions.to(dtype=torch.long).argsort(dim=1, descending=True, stable=True)
            compact = valid_sequence.gather(1, order.unsqueeze(-1).expand(-1, -1, 1))
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
