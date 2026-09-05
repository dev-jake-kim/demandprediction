"""another_model-style local demand history encoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .embeddings import FourierScalarEmbedding


class LocalHistoryEncoder(nn.Module):
    """Local crop -> Transformer per history step -> temporal LSTM."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        time_step: int,
        local_radius: int,
        d_model: int,
        num_fourier_bands: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ffn: int,
        history_hidden: int,
        dropout: float,
        extra_dim: int = 0,
        weather_cls_dim: int = 0,
        use_neighbors: bool = True,
    ) -> None:
        super().__init__()
        if extra_dim < 0:
            raise ValueError("extra_dim must be non-negative")
        if weather_cls_dim < 0:
            raise ValueError("weather_cls_dim must be non-negative")
        # >0이면 ir-weather 방식 — 정규화 날씨를 d_model로 투영해 CLS 토큰에 더한다.
        # LSTM 입력 concat과 배타적으로 쓴다(둘 다 켜면 같은 정보가 두 경로로 들어간다).
        # False면 (2a+1)^2 창에서 중앙(자기 노드)만 남기고 나머지 이웃 토큰을 0으로 바꾼다.
        # 토큰 개수와 파라미터는 그대로라 '공간 이웃 정보'만 제거된다.
        self.use_neighbors = use_neighbors
        self.weather_cls_dim = weather_cls_dim
        self.weather_projection = (
            nn.Linear(weather_cls_dim, d_model) if weather_cls_dim > 0 else None
        )
        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = time_step
        self.local_radius = local_radius
        self.window_size = 2 * local_radius + 1
        self.num_neighbors = self.window_size * self.window_size
        self.extra_dim = extra_dim

        self.register_buffer("neighbor_valid", self._make_neighbor_valid(), persistent=False)
        self.scalar_embedding = FourierScalarEmbedding(d_model, num_fourier_bands)
        self.special_embedding = nn.Embedding(2, d_model)  # 0: CLS, 1: EDGE
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1 + self.num_neighbors, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.history_lstm = nn.LSTM(d_model + extra_dim, history_hidden, batch_first=True)

    def _make_neighbor_valid(self) -> Tensor:
        valid = np.zeros((self.num_nodes, self.num_neighbors), dtype=bool)
        column = 0
        for dy in range(-self.local_radius, self.local_radius + 1):
            for dx in range(-self.local_radius, self.local_radius + 1):
                for node in range(self.num_nodes):
                    y, x = divmod(node, self.width)
                    valid[node, column] = (
                        0 <= y + dy < self.height and 0 <= x + dx < self.width
                    )
                column += 1
        return torch.from_numpy(valid)

    def crop(self, demands: Tensor) -> Tensor:
        """Return raw local windows as ``[B, k, N, neighbors]``."""

        if demands.ndim != 4 or tuple(demands.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                f"demand_history must be [B,k,{self.height},{self.width}], got {tuple(demands.shape)}"
            )
        batch, steps = demands.shape[:2]
        padded = F.pad(
            demands.reshape(batch * steps, 1, self.height, self.width),
            (self.local_radius,) * 4,
        )
        patches = F.unfold(padded, kernel_size=self.window_size)
        return patches.transpose(1, 2).reshape(batch, steps, self.num_nodes, self.num_neighbors)

    def forward(
        self,
        demands: Tensor,
        extra: Tensor | None = None,
        weather_cls: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        local_crop = self.crop(demands)
        batch, steps, nodes, neighbors = local_crop.shape
        if steps != self.time_step or nodes != self.num_nodes or neighbors != self.num_neighbors:
            raise ValueError("Unexpected local crop shape")
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError(f"extra_dim={self.extra_dim}인데 extra가 None임")
            if extra.shape != (batch, steps, self.extra_dim):
                raise ValueError(f"extra must be [B,k,{self.extra_dim}], got {tuple(extra.shape)}")
        elif extra is not None:
            raise ValueError("extra_dim=0인데 extra가 주어짐")

        valid = self.neighbor_valid.to(device=local_crop.device)
        log_values = torch.log1p(torch.clamp(local_crop, min=0.0)).unsqueeze(-1)
        value_tokens = self.scalar_embedding(log_values)
        edge_token = self.special_embedding.weight[1].view(1, 1, 1, 1, -1)
        value_tokens = torch.where(valid.view(1, 1, nodes, neighbors, 1), value_tokens, edge_token)

        if not self.use_neighbors:
            # EDGE 치환까지 끝난 뒤에 0으로 만든다 — 격자 밖 이웃도 함께 0이 되어야
            # "중앙 외 공간 정보 없음"이 균일하게 적용된다. unfold는 행 우선이라 홀수 창의
            # 중앙은 항상 neighbors // 2다(a=2면 25개 중 12번).
            keep = torch.zeros(neighbors, dtype=value_tokens.dtype, device=value_tokens.device)
            keep[neighbors // 2] = 1.0
            value_tokens = value_tokens * keep.view(1, 1, 1, neighbors, 1)

        cls_token = self.special_embedding.weight[0].view(1, 1, 1, 1, -1)
        cls_token = cls_token + self.node_embedding.view(1, 1, nodes, 1, -1)
        cls_token = cls_token.expand(batch, steps, -1, -1, -1)
        if self.weather_projection is not None:
            if weather_cls is None:
                raise ValueError("weather_cls_dim>0인데 weather_cls가 None임")
            # 날씨는 노드에 무관하므로 [B,k,d]를 노드/이웃 축으로 브로드캐스트해 CLS에만 더한다.
            cls_token = cls_token + self.weather_projection(weather_cls)[:, :, None, None, :]
        tokens = torch.cat([cls_token, value_tokens], dim=3)
        tokens = tokens + self.position_embedding.view(1, 1, 1, 1 + neighbors, -1)
        encoded = self.transformer(tokens.reshape(batch * steps * nodes, 1 + neighbors, -1))
        cls = encoded[:, 0].reshape(batch, steps, nodes, -1)
        sequence = cls.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        # 날씨/캘린더는 노드에 무관하므로 [B,k,E]를 노드 축으로 브로드캐스트해 LSTM 입력에 붙인다.
        if self.extra_dim > 0:
            expanded_extra = (
                extra[:, None, :, :]
                .expand(batch, nodes, steps, self.extra_dim)
                .reshape(batch * nodes, steps, self.extra_dim)
            )
            sequence = torch.cat([sequence, expanded_extra], dim=-1)

        _, (hidden, _) = self.history_lstm(sequence)
        return local_crop, hidden[-1].reshape(batch, nodes, -1)


__all__ = ["LocalHistoryEncoder"]
