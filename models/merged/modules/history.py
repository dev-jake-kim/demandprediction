"""Local demand history encoder."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .embeddings import FourierScalarEmbedding


class LocalHistoryEncoder(nn.Module):
    """전체 노드 토큰 + 층별 sink -> 이웃 창 masked Transformer (시점별) -> temporal LSTM."""

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
        self.transformer_heads = transformer_heads
        self.extra_dim = extra_dim

        # from_pretrained가 체크포인트에서 복원하도록 persistent 버퍼로 둔다.
        self.register_buffer("neighbor_direction", self._make_neighbor_direction(), persistent=True)
        self.scalar_embedding = FourierScalarEmbedding(d_model, num_fourier_bands)
        # 층마다 독립 sink 토큰. 한 층의 sink 출력은 다음 층으로 전달하지 않는다.
        self.sink_embedding = nn.Parameter(torch.randn(transformer_layers, d_model) * 0.02)
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.direction_bias = nn.Parameter(torch.zeros(transformer_heads, self.num_neighbors))

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
        self.transformer = nn.ModuleList(
            copy.deepcopy(encoder_layer) for _ in range(transformer_layers)
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

    def _make_neighbor_direction(self) -> Tensor:
        """``[N, N]`` long. query 노드 기준 key 노드의 창 내 방향 index, 창 밖이면 -1.

        방향 index는 ``(dy + r) * window + (dx + r)``(행 우선)이며 자기 자신은 ``P // 2``다.
        """

        radius = self.local_radius
        node = torch.arange(self.num_nodes)
        y = torch.div(node, self.width, rounding_mode="floor")
        x = node % self.width
        dy = y.unsqueeze(0) - y.unsqueeze(1)
        dx = x.unsqueeze(0) - x.unsqueeze(1)
        inside = (dy.abs() <= radius) & (dx.abs() <= radius)
        if not self.use_neighbors:
            inside = inside & (dy == 0) & (dx == 0)
        direction = (dy + radius) * self.window_size + (dx + radius)
        return torch.where(inside, direction, torch.full_like(direction, -1))

    def _attention_mask(self, sequences: int) -> Tensor:
        """``[sequences * heads, 1+N, 1+N]`` 더하는 attention mask (index 0 = sink).

        sink 행·열은 막지 않는다. 노드 쌍은 창 안이면 방향별·head별 bias, 밖이면 -inf.
        """

        direction = self.neighbor_direction
        allowed = F.pad(direction >= 0, (1, 0, 1, 0), value=True)
        bias = F.pad(self.direction_bias[:, direction.clamp_min(0)], (1, 0, 1, 0))
        mask = bias.masked_fill(~allowed, float("-inf"))
        length = mask.shape[-1]
        return (
            mask.unsqueeze(0)
            .expand(sequences, -1, -1, -1)
            .reshape(sequences * self.transformer_heads, length, length)
        )

    def forward(
        self,
        demands: Tensor,
        extra: Tensor | None = None,
        weather_cls: Tensor | None = None,
    ) -> Tensor:
        """``[B, k, H, W]`` 수요 -> ``[B, N, history_hidden]``."""

        if demands.ndim != 4 or tuple(demands.shape[1:]) != (self.time_step, self.height, self.width):
            raise ValueError(
                f"demand_history must be [B,{self.time_step},{self.height},{self.width}], "
                f"got {tuple(demands.shape)}"
            )
        batch, steps = demands.shape[:2]
        nodes = self.num_nodes
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError(f"extra_dim={self.extra_dim}인데 extra가 None임")
            if extra.shape != (batch, steps, self.extra_dim):
                raise ValueError(f"extra must be [B,k,{self.extra_dim}], got {tuple(extra.shape)}")
        elif extra is not None:
            raise ValueError("extra_dim=0인데 extra가 주어짐")

        log_values = torch.log1p(torch.clamp(demands, min=0.0)).reshape(batch, steps, nodes, 1)
        tokens = self.scalar_embedding(log_values) + self.node_embedding
        if self.weather_projection is not None:
            if weather_cls is None:
                raise ValueError("weather_cls_dim>0인데 weather_cls가 None임")
            tokens = tokens + self.weather_projection(weather_cls)[:, :, None, :]
        sequences = batch * steps
        hidden_tokens = tokens.reshape(sequences, nodes, -1)
        mask = self._attention_mask(sequences)
        # eval·no_grad fast path는 head별 3D float mask를 무시한 결과를 내므로 끈다.
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for layer, sink in zip(self.transformer, self.sink_embedding):
                layer_input = torch.cat([sink.expand(sequences, 1, -1), hidden_tokens], dim=1)
                hidden_tokens = layer(layer_input, src_mask=mask)[:, 1:]
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        summary = hidden_tokens.reshape(batch, steps, nodes, -1)
        sequence = summary.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

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

        return node_hidden


__all__ = ["LocalHistoryEncoder"]
