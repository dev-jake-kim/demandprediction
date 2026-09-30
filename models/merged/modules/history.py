"""Local demand history encoder."""

from __future__ import annotations

from collections.abc import Sequence

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
        attention_dropout: float | None = None,
        extra_dim: int = 0,
        weather_cls_dim: int = 0,
        use_neighbors: bool = True,
        node_adaptive_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if extra_dim < 0:
            raise ValueError("extra_dim must be non-negative")
        if weather_cls_dim < 0:
            raise ValueError("weather_cls_dim must be non-negative")
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

        # from_pretrained가 체크포인트에서 복원하도록 persistent 버퍼로 둔다.
        self.register_buffer("neighbor_valid", self._make_neighbor_valid(), persistent=True)
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
        encoder_layer.self_attn.dropout = (
            dropout if attention_dropout is None else attention_dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.history_lstm = nn.LSTM(d_model + extra_dim, history_hidden, batch_first=True)
        self.history_hidden = history_hidden

        self.node_adaptive = node_adaptive_indices is not None
        if self.node_adaptive:
            # meta device 초기화에서도 검사할 수 있도록 파이썬 리스트로 다룬다.
            index_list = [int(value) for value in node_adaptive_indices]
            if not index_list:
                raise ValueError("node_adaptive_indices는 비어 있지 않은 노드 id 목록이어야 함")
            if min(index_list) < 0 or max(index_list) >= self.num_nodes:
                raise ValueError(
                    f"node_adaptive_indices가 노드 범위(0..{self.num_nodes - 1})를 벗어남"
                )
            if len(set(index_list)) != len(index_list):
                raise ValueError("node_adaptive_indices에 중복이 있음")
            num_adaptive = len(index_list)
            n_gates = 4 * history_hidden
            # 노드 id는 버퍼가 아니라 리스트로 보관한다: 체크포인트에 없는 버퍼는
            # from_pretrained에서 torch.empty 값으로 남는다. 텐서는 device별로 캐시한다.
            self._node_adaptive_index_list = index_list
            self._cached_node_index: Tensor | None = None
            self.node_delta_weight_ih = nn.Parameter(
                torch.zeros(num_adaptive, n_gates, d_model + extra_dim)
            )
            self.node_delta_weight_hh = nn.Parameter(
                torch.zeros(num_adaptive, n_gates, history_hidden)
            )
            # One delta bias offsets the combined ``bias_ih + bias_hh``.
            self.node_delta_bias = nn.Parameter(torch.zeros(num_adaptive, n_gates))

    def node_adaptive_index(self, device: torch.device) -> Tensor:
        """ΔW를 받는 노드 id 텐서. config에서 온 파이썬 리스트로 매번 만들되 device별로 캐시한다."""

        cached = self._cached_node_index
        if cached is None or cached.device != device:
            cached = torch.as_tensor(
                self._node_adaptive_index_list, dtype=torch.long, device=device
            )
            self._cached_node_index = cached
        return cached

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        """노드별 weight offset 파라미터 전부. 초기화/동결/검증에서 한 곳으로 모아 쓴다."""

        if not self.node_adaptive:
            return ()
        return (
            self.node_delta_weight_ih,
            self.node_delta_weight_hh,
            self.node_delta_bias,
        )

    def _node_adaptive_hidden(self, sequence: Tensor) -> Tensor:
        """Compute adaptive-node hidden states with per-node LSTM offsets.

        ``sequence`` has shape ``[B, A, k, d_model + extra_dim]`` and the
        returned tensor has shape ``[B, A, history_hidden]``.
        """

        batch, num_adaptive, steps, _ = sequence.shape
        weight_ih = self.history_lstm.weight_ih_l0 + self.node_delta_weight_ih  # (A,4h,F)
        weight_hh = self.history_lstm.weight_hh_l0 + self.node_delta_weight_hh  # (A,4h,h)
        bias = (
            self.history_lstm.bias_ih_l0 + self.history_lstm.bias_hh_l0 + self.node_delta_bias
        )  # (A,4h)

        gates_x = torch.einsum("bakf,agf->bakg", sequence, weight_ih) + bias.unsqueeze(1)

        hidden = sequence.new_zeros(batch, num_adaptive, self.history_hidden)
        cell = sequence.new_zeros(batch, num_adaptive, self.history_hidden)
        for step in range(steps):
            gates = gates_x[:, :, step] + torch.einsum("bah,agh->bag", hidden, weight_hh)
            in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=-1)
            # PyTorch LSTM 게이트 순서: i, f, g, o
            cell = forget_gate.sigmoid() * cell + in_gate.sigmoid() * cell_gate.tanh()
            hidden = out_gate.sigmoid() * cell.tanh()
        return hidden

    def _make_neighbor_valid(self) -> Tensor:
        # Window columns use row-major offsets: dy = c // window - r, dx = c % window - r.
        radius, window = self.local_radius, self.window_size
        offsets = torch.arange(-radius, radius + 1)
        dy = offsets.repeat_interleave(window)  # [neighbors]
        dx = offsets.repeat(window)  # [neighbors]
        node = torch.arange(self.num_nodes)
        y = torch.div(node, self.width, rounding_mode="floor").unsqueeze(1) + dy.unsqueeze(0)
        x = (node % self.width).unsqueeze(1) + dx.unsqueeze(0)
        return (y >= 0) & (y < self.height) & (x >= 0) & (x < self.width)

    def crop(self, demands: Tensor, radius: int | None = None) -> Tensor:
        """Return raw local windows as ``[B, k, N, (2*radius+1)^2]``."""

        if demands.ndim != 4 or tuple(demands.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                f"demand_history must be [B,k,{self.height},{self.width}], got {tuple(demands.shape)}"
            )
        radius = self.local_radius if radius is None else radius
        window_size = 2 * radius + 1
        batch, steps = demands.shape[:2]
        padded = F.pad(
            demands.reshape(batch * steps, 1, self.height, self.width),
            (radius,) * 4,
        )
        patches = F.unfold(padded, kernel_size=window_size)
        return patches.transpose(1, 2).reshape(batch, steps, self.num_nodes, window_size ** 2)

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
            # Zero after EDGE replacement so all non-central tokens are disabled uniformly.
            keep = torch.zeros(neighbors, dtype=value_tokens.dtype, device=value_tokens.device)
            keep[neighbors // 2] = 1.0
            value_tokens = value_tokens * keep.view(1, 1, 1, neighbors, 1)

        cls_token = self.special_embedding.weight[0].view(1, 1, 1, 1, -1)
        cls_token = cls_token + self.node_embedding.view(1, 1, nodes, 1, -1)
        cls_token = cls_token.expand(batch, steps, -1, -1, -1)
        if self.weather_projection is not None:
            if weather_cls is None:
                raise ValueError("weather_cls_dim>0인데 weather_cls가 None임")
            cls_token = cls_token + self.weather_projection(weather_cls)[:, :, None, None, :]
        tokens = torch.cat([cls_token, value_tokens], dim=3)
        tokens = tokens + self.position_embedding.view(1, 1, 1, 1 + neighbors, -1)
        encoded = self.transformer(tokens.reshape(batch * steps * nodes, 1 + neighbors, -1))
        cls = encoded[:, 0].reshape(batch, steps, nodes, -1)
        sequence = cls.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        if self.extra_dim > 0:
            expanded_extra = (
                extra[:, None, :, :]
                .expand(batch, nodes, steps, self.extra_dim)
                .reshape(batch * nodes, steps, self.extra_dim)
            )
            sequence = torch.cat([sequence, expanded_extra], dim=-1)

        _, (hidden, _) = self.history_lstm(sequence)
        node_hidden = hidden[-1].reshape(batch, nodes, -1)

        if self.node_adaptive:
            index = self.node_adaptive_index(sequence.device)
            adaptive_sequence = sequence.reshape(batch, nodes, steps, -1).index_select(1, index)
            adaptive_hidden = self._node_adaptive_hidden(adaptive_sequence)
            # Use out-of-place index_copy so both paths remain differentiable.
            node_hidden = node_hidden.index_copy(1, index, adaptive_hidden)

        return local_crop, node_hidden


__all__ = ["LocalHistoryEncoder"]
