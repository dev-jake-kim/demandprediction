"""Daily and weekly closed-form trend forecasters."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearTrendForecaster(nn.Module):
    """lag 수열의 최소제곱 직선을 다음 위치로 외삽한다(무효 lag가 있으면 유효 lag 평균)."""

    def forward(self, values: Tensor, invalid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """``[B,L,N,1]`` lag 수요(오래된 것부터), 무효=True인 ``[B,L]`` -> (``[B,N]`` 예측, ``[B]`` 유효)."""

        if values.ndim != 4 or values.shape[-1] != 1:
            raise ValueError(f"Periodic values must be [B,L,N,1], got {tuple(values.shape)}")
        if invalid_mask.ndim != 2 or invalid_mask.shape[:2] != values.shape[:2]:
            raise ValueError("Periodic mask must be [B,L] matching the first two value dimensions")
        length = values.shape[1]
        if length < 2:
            raise ValueError(f"lag가 2개 이상이어야 직선을 맞출 수 있음: L={length}")

        y = values.squeeze(-1)  # [B,L,N]
        valid_mask = ~invalid_mask.bool()
        counts = valid_mask.sum(dim=1)  # [B]

        position = torch.arange(1, length + 1, dtype=y.dtype, device=y.device)
        centered = position - position.mean()
        y_mean = y.mean(dim=1)  # [B,N]
        slope = torch.einsum('l,bln->bn', centered, y) / centered.pow(2).sum()
        trend = y_mean + slope * (length + 1 - position.mean())

        weights = valid_mask.to(y.dtype)
        valid_mean = torch.einsum('bl,bln->bn', weights, y) / counts.clamp_min(1)[:, None].to(y.dtype)

        all_valid = (counts == length)[:, None]
        prediction = torch.where(all_valid, trend, valid_mean).clamp_min(0.0)
        valid = counts > 0
        return prediction * valid[:, None].to(y.dtype), valid


__all__ = ["LinearTrendForecaster"]
