"""Local demand history encoder."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .embeddings import FourierScalarEmbedding
from .node_delta import NodeLSTMMoE, run_dynamic_basis_lstm, run_node_lstm


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
        use_sink: bool = True,
        local_encoder: str = "masked_grid",
        global_nodes: Sequence[int] | None = None,
        node_adaptive_indices: Sequence[int] | None = None,
        node_adaptive_mode: str = "additive",
        functional_neighbors: Sequence[Sequence[int]] | None = None,
        use_event_embedding: bool = False,
        aux_channels: int = 0,
        node_delta_rank: int = 8,
        node_descriptors: Sequence[Sequence[float]] | None = None,
        node_delta_dynamic: bool = False,
        node_moe_experts: int = 4,
        node_moe_router: str = "static",
        node_moe_top_k: int = 0,
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
        if local_encoder not in ("masked_grid", "crop"):
            raise ValueError(f"local_encoder는 'masked_grid'|'crop' (받음: {local_encoder!r})")
        self.local_encoder = local_encoder
        # crop(이전 방식, ablation 전용)은 sink·방향 bias 대신 CLS/EDGE 토큰과 창 위치 임베딩을 쓴다.
        self.use_sink = use_sink and local_encoder == "masked_grid"
        # 층마다 독립 sink 토큰. 한 층의 sink 출력은 다음 층으로 전달하지 않는다. use_sink=false면 없다.
        if self.use_sink:
            self.sink_embedding = nn.Parameter(torch.randn(transformer_layers, d_model) * 0.02)
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        if local_encoder == "masked_grid":
            self.direction_bias = nn.Parameter(torch.zeros(transformer_heads, self.num_neighbors))
            # 모든 노드가 볼 수 있는 노드(마스크 열 개방). 창 밖 쌍은 head별 global_bias(0 초기화)를 쓴다.
            self._global_node_list = sorted({int(v) for v in (global_nodes or [])})
            if self._global_node_list:
                if min(self._global_node_list) < 0 or max(self._global_node_list) >= self.num_nodes:
                    raise ValueError(f"global_nodes가 노드 범위(0..{self.num_nodes - 1})를 벗어남")
                self.global_bias = nn.Parameter(torch.zeros(transformer_heads))
        else:
            if functional_neighbors is not None or use_event_embedding or weather_cls_dim > 0 or global_nodes:
                raise ValueError("local_encoder=crop은 기능적 이웃·이벤트 임베딩·cls_add 날씨·global_nodes를 지원하지 않음")
            self.register_buffer("neighbor_valid", self._make_neighbor_valid(), persistent=True)
            self.special_embedding = nn.Embedding(2, d_model)  # 0: CLS, 1: EDGE(격자 밖)
            self.position_embedding = nn.Parameter(torch.randn(1 + self.num_neighbors, d_model) * 0.02)
        # 모든 이벤트(공연·축제 등)가 공유하는 학습 임베딩. 0 초기화라 학습 전에는 없는 것과 같다.
        self.use_event_embedding = use_event_embedding
        if use_event_embedding:
            self.event_embedding = nn.Parameter(torch.zeros(d_model))
        # 기능적 이웃: 노드마다 train 구간 상관이 큰 먼 노드 K개. 그 최근 수요 평균을 토큰에 더한다.
        self._functional_neighbor_list = (
            None if functional_neighbors is None else [[int(v) for v in row] for row in functional_neighbors]
        )
        self._cached_functional_index: Tensor | None = None
        if self._functional_neighbor_list is not None:
            if len(self._functional_neighbor_list) != self.num_nodes:
                raise ValueError('functional_neighbors는 노드마다 한 행이어야 함')
            self.functional_projection = nn.Linear(1, d_model)
        # 노드별 배차 상태(실패·대기 등) C채널을 log1p 후 Linear(C→D)로 토큰에 더한다(선택).
        self.aux_channels = int(aux_channels)
        if self.aux_channels > 0:
            if local_encoder != "masked_grid":
                raise ValueError("aux_channels는 masked_grid 인코더에서만 지원")
            self.aux_projection = nn.Linear(self.aux_channels, d_model)

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
        if node_adaptive_mode not in ("additive", "gated", "basis", "moe"):
            raise ValueError(f"node_adaptive_mode는 'additive'|'gated'|'basis'|'moe' (받음: {node_adaptive_mode!r})")
        self.node_adaptive_mode = node_adaptive_mode
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
            if node_adaptive_mode == "basis":
                # basis: ΔW[n] = Σ_m c[n,m]·B[m], c[n] = C[n] (+ φ[n]·W_φ). 전부 0으로 생성, begin_node_adaptation이 C·W_φ를 채운다.
                rank = int(node_delta_rank)
                if rank < 1:
                    raise ValueError("node_delta_rank는 1 이상이어야 함")
                self.node_delta_rank = rank
                self.node_delta_basis_ih = nn.Parameter(torch.zeros(rank, n_gates, d_model + extra_dim))
                self.node_delta_basis_hh = nn.Parameter(torch.zeros(rank, n_gates, history_hidden))
                self.node_delta_basis_bias = nn.Parameter(torch.zeros(rank, n_gates))
                self.node_delta_coef = nn.Parameter(torch.zeros(num_adaptive, rank))
                self.node_delta_descriptor_weight = None
                if node_descriptors is not None:
                    # 버퍼 대신 리스트(meta device 초기화·from_pretrained 안전). 텐서는 device별 캐시.
                    self._node_descriptor_list = [[float(v) for v in row] for row in node_descriptors]
                    widths = {len(row) for row in self._node_descriptor_list}
                    if len(self._node_descriptor_list) != num_adaptive or len(widths) != 1:
                        raise ValueError(f"node_descriptors는 [A={num_adaptive}, F] 이어야 함")
                    self._cached_descriptor_table: Tensor | None = None
                    self.node_delta_descriptor_weight = nn.Parameter(torch.zeros(widths.pop(), rank))
                # dynamic: 샘플·노드별 계수 c[b,n] = c[n] ⊙ (1 + tanh(x_last[b,n]·W_dyn)), W_dyn 0 시작(정적 basis와 같음).
                self.node_delta_dynamic_weight = (
                    nn.Parameter(torch.zeros(d_model + extra_dim, rank)) if node_delta_dynamic else None
                )
            elif node_delta_dynamic:
                raise ValueError("node_delta_dynamic은 node_adaptive_mode=basis에서만 쓴다")
            elif node_adaptive_mode == "moe":
                if node_descriptors is not None:
                    raise ValueError("node_descriptors는 node_adaptive_mode=basis에서만 쓴다")
                # moe: 공유 전문가 K개(W + ΔW_k)를 노드별 router로 섞는다(docs/SPEC.md §6.7).
                self.node_moe = NodeLSTMMoE(
                    num_adaptive, d_model + extra_dim, history_hidden,
                    node_moe_experts, node_moe_router, node_moe_top_k,
                )
            else:
                if node_descriptors is not None:
                    raise ValueError("node_descriptors는 node_adaptive_mode=basis에서만 쓴다")
                self.node_delta_weight_ih = nn.Parameter(
                    torch.zeros(num_adaptive, n_gates, d_model + extra_dim)
                )
                self.node_delta_weight_hh = nn.Parameter(
                    torch.zeros(num_adaptive, n_gates, history_hidden)
                )
                # One delta bias offsets the combined ``bias_ih + bias_hh``.
                self.node_delta_bias = nn.Parameter(torch.zeros(num_adaptive, n_gates))
            if node_adaptive_mode == "gated":
                # gated: node_delta_*는 노드 고유 가중치 V[n]. g[n] = sigmoid(logit[n]).
                self.node_delta_gate_logit = nn.Parameter(torch.zeros(num_adaptive))
                # s: 0이면 공유 가중치와 같은 함수(stage 1). begin_node_adaptation이 1로 둔다.
                self.register_buffer("node_delta_started", torch.zeros(()), persistent=True)

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
        if self.node_adaptive_mode == "moe":
            return self.node_moe.node_delta_parameters()
        if self.node_adaptive_mode == "basis":
            deltas = (
                self.node_delta_basis_ih,
                self.node_delta_basis_hh,
                self.node_delta_basis_bias,
                self.node_delta_coef,
            )
            if self.node_delta_descriptor_weight is not None:
                deltas += (self.node_delta_descriptor_weight,)
            if self.node_delta_dynamic_weight is not None:
                deltas += (self.node_delta_dynamic_weight,)
            return deltas
        deltas = (
            self.node_delta_weight_ih,
            self.node_delta_weight_hh,
            self.node_delta_bias,
        )
        if self.node_adaptive_mode == "gated":
            deltas += (self.node_delta_gate_logit,)
        return deltas

    def node_delta_coefficients(self) -> Tensor:
        """basis 모드의 노드별 결합 계수 c[n] ``[A, M]``."""

        coef = self.node_delta_coef
        if self.node_delta_descriptor_weight is not None:
            table = self._cached_descriptor_table
            if table is None or table.device != coef.device:
                table = torch.as_tensor(self._node_descriptor_list, dtype=torch.float32, device=coef.device)
                self._cached_descriptor_table = table
            coef = coef + table.to(coef.dtype) @ self.node_delta_descriptor_weight
        return coef

    @torch.no_grad()
    def begin_node_adaptation(self) -> None:
        """ΔW 학습 시작 직전 호출(출력 불변). gated: V[n] ← 현재 공유 가중치, s ← 1.
        basis: 기저 B=0을 유지하고 계수 C(·W_φ)만 난수로 채운다. moe: router 표만 난수로 채운다."""

        if not self.node_adaptive:
            return
        if self.node_adaptive_mode == "moe":
            self.node_moe.begin()
            return
        if self.node_adaptive_mode == "basis":
            rank = self.node_delta_rank
            if self.node_delta_descriptor_weight is None:
                self.node_delta_coef.normal_(0.0, (1.0 / rank) ** 0.5)
            else:
                features = self.node_delta_descriptor_weight.shape[0]
                self.node_delta_coef.normal_(0.0, (0.5 / rank) ** 0.5)
                self.node_delta_descriptor_weight.normal_(0.0, (0.5 / (rank * features)) ** 0.5)
            return
        if self.node_adaptive_mode != "gated":
            return
        lstm = self.history_lstm
        self.node_delta_weight_ih.copy_(lstm.weight_ih_l0.expand_as(self.node_delta_weight_ih))
        self.node_delta_weight_hh.copy_(lstm.weight_hh_l0.expand_as(self.node_delta_weight_hh))
        self.node_delta_bias.copy_(
            (lstm.bias_ih_l0 + lstm.bias_hh_l0).expand_as(self.node_delta_bias)
        )
        self.node_delta_started.fill_(1.0)

    def node_gate(self) -> Tensor | None:
        """gated면 노드별 g[n] = sigmoid(logit) ``[A]``, 아니면 None."""

        if not self.node_adaptive or self.node_adaptive_mode != "gated":
            return None
        return self.node_delta_gate_logit.sigmoid()

    def _node_adaptive_hidden(self, sequence: Tensor) -> Tensor:
        """Compute adaptive-node hidden states with per-node LSTM offsets.

        ``sequence`` has shape ``[B, A, k, d_model + extra_dim]`` and the
        returned tensor has shape ``[B, A, history_hidden]``.
        """

        lstm = self.history_lstm
        if self.node_adaptive_mode == "moe":
            return self.node_moe(lstm, sequence)[:, :, -1]
        shared_bias = lstm.bias_ih_l0 + lstm.bias_hh_l0
        if self.node_adaptive_mode == "gated":
            # W + s·(1−g)·(V − W); s=1이면 g·W + (1−g)·V.
            mix = self.node_delta_started * (1.0 - self.node_delta_gate_logit.sigmoid())  # (A,)
            weight_ih = lstm.weight_ih_l0 + mix[:, None, None] * (self.node_delta_weight_ih - lstm.weight_ih_l0)
            weight_hh = lstm.weight_hh_l0 + mix[:, None, None] * (self.node_delta_weight_hh - lstm.weight_hh_l0)
            bias = shared_bias + mix[:, None] * (self.node_delta_bias - shared_bias)
        elif self.node_adaptive_mode == "basis":
            coef = self.node_delta_coefficients()  # (A,M)
            if self.node_delta_dynamic_weight is not None:
                # 마지막 시점 입력(Transformer 토큰 + 시각·날씨 context)으로 노드 계수를 샘플마다 조절한다.
                scale = 1.0 + torch.tanh(sequence[:, :, -1] @ self.node_delta_dynamic_weight)  # (B,A,M)
                return run_dynamic_basis_lstm(
                    sequence, lstm, coef[None] * scale,
                    self.node_delta_basis_ih, self.node_delta_basis_hh, self.node_delta_basis_bias,
                )
            weight_ih = lstm.weight_ih_l0 + torch.einsum("am,mgf->agf", coef, self.node_delta_basis_ih)
            weight_hh = lstm.weight_hh_l0 + torch.einsum("am,mgh->agh", coef, self.node_delta_basis_hh)
            bias = shared_bias + coef @ self.node_delta_basis_bias
        else:
            weight_ih = lstm.weight_ih_l0 + self.node_delta_weight_ih  # (A,4h,F)
            weight_hh = lstm.weight_hh_l0 + self.node_delta_weight_hh  # (A,4h,h)
            bias = shared_bias + self.node_delta_bias  # (A,4h)

        return run_node_lstm(sequence, weight_ih, weight_hh, bias)[:, :, -1]

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
        """더하는 attention mask. use_sink면 ``[sequences*heads, 1+N, 1+N]`` (index 0 = sink), 아니면 ``[.., N, N]``.

        sink 행·열은 막지 않는다. 노드 쌍은 창 안이면 방향별·head별 bias, 밖이면 -inf.
        """

        direction = self.neighbor_direction
        allowed = direction >= 0
        bias = self.direction_bias[:, direction.clamp_min(0)]
        if self._global_node_list:
            column = torch.zeros_like(allowed)
            column[:, self._global_node_list] = True
            extra = column & ~allowed  # 창 밖에서 새로 열리는 (i, g) 쌍
            bias = torch.where(extra, self.global_bias[:, None, None], bias)
            allowed = allowed | column
        if self.use_sink:
            allowed = F.pad(allowed, (1, 0, 1, 0), value=True)
            bias = F.pad(bias, (1, 0, 1, 0))
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
        event_mask: Tensor | None = None,
        aux_history: Tensor | None = None,
    ) -> Tensor:
        """``[B, k, H, W]`` 수요 -> ``[B, N, history_hidden]``. ``event_mask``는 ``[B, k, N]`` 0/1, ``aux_history``는 ``[B, k, N, C]``."""

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

        if self.local_encoder == "crop":
            if weather_cls is not None or event_mask is not None or aux_history is not None:
                raise ValueError("local_encoder=crop은 weather_cls·event_mask·aux_history를 받지 않음")
            summary = self._crop_summary(demands)
        else:
            summary = self._masked_grid_summary(demands, weather_cls, event_mask, aux_history)
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

    def _masked_grid_summary(
        self, demands: Tensor, weather_cls: Tensor | None, event_mask: Tensor | None,
        aux_history: Tensor | None = None,
    ) -> Tensor:
        """전체 노드 토큰 + 층별 sink -> 이웃 창 masked Transformer. ``[B, k, N, D]``."""

        batch, steps = demands.shape[:2]
        nodes = self.num_nodes
        log_values = torch.log1p(torch.clamp(demands, min=0.0)).reshape(batch, steps, nodes, 1)
        tokens = self.scalar_embedding(log_values) + self.node_embedding
        if self._functional_neighbor_list is not None:
            index = self._cached_functional_index
            if index is None or index.device != demands.device:
                index = torch.as_tensor(self._functional_neighbor_list, dtype=torch.long, device=demands.device)
                self._cached_functional_index = index
            flat = torch.clamp(demands, min=0.0).reshape(batch, steps, nodes)
            neighbor_mean = flat[:, :, index].mean(dim=-1)  # [B, k, N]
            tokens = tokens + self.functional_projection(torch.log1p(neighbor_mean).unsqueeze(-1))
        if self.weather_projection is not None:
            if weather_cls is None:
                raise ValueError("weather_cls_dim>0인데 weather_cls가 None임")
            tokens = tokens + self.weather_projection(weather_cls)[:, :, None, :]
        if self.use_event_embedding:
            if event_mask is None or tuple(event_mask.shape) != (batch, steps, nodes):
                raise ValueError(f"use_event_embedding이면 event_mask [B,{steps},{nodes}]가 필요함")
            tokens = tokens + event_mask.to(tokens.dtype).unsqueeze(-1) * self.event_embedding
        elif event_mask is not None:
            raise ValueError("use_event_embedding=false인데 event_mask가 주어짐")
        if self.aux_channels > 0:
            if aux_history is None or tuple(aux_history.shape) != (batch, steps, nodes, self.aux_channels):
                raise ValueError(f"aux_channels={self.aux_channels}이면 aux_history [B,{steps},{nodes},{self.aux_channels}]가 필요함")
            tokens = tokens + self.aux_projection(torch.log1p(aux_history.to(tokens.dtype)))
        elif aux_history is not None:
            raise ValueError("aux_channels=0인데 aux_history가 주어짐")
        sequences = batch * steps
        hidden_tokens = tokens.reshape(sequences, nodes, -1)
        mask = self._attention_mask(sequences)
        # eval·no_grad fast path는 head별 3D float mask를 무시한 결과를 내므로 끈다.
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for index, layer in enumerate(self.transformer):
                if self.use_sink:
                    sink = self.sink_embedding[index].expand(sequences, 1, -1)
                    hidden_tokens = layer(torch.cat([sink, hidden_tokens], dim=1), src_mask=mask)[:, 1:]
                else:
                    hidden_tokens = layer(hidden_tokens, src_mask=mask)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        return hidden_tokens.reshape(batch, steps, nodes, -1)

    def _make_neighbor_valid(self) -> Tensor:
        """``[N, P]`` bool: 창 위치(행 우선 ``dy, dx``)가 격자 안이면 True (crop 전용)."""

        radius, window = self.local_radius, self.window_size
        offsets = torch.arange(-radius, radius + 1)
        dy = offsets.repeat_interleave(window)
        dx = offsets.repeat(window)
        node = torch.arange(self.num_nodes)
        y = torch.div(node, self.width, rounding_mode="floor").unsqueeze(1) + dy.unsqueeze(0)
        x = (node % self.width).unsqueeze(1) + dx.unsqueeze(0)
        return (y >= 0) & (y < self.height) & (x >= 0) & (x < self.width)

    def _crop_summary(self, demands: Tensor) -> Tensor:
        """이전 방식: 노드별 창을 잘라 ``[CLS+node_emb, 창 값 토큰] + 위치 임베딩``을 독립 Transformer에 넣고 CLS 출력. ``[B, k, N, D]``."""

        batch, steps = demands.shape[:2]
        nodes, neighbors, radius = self.num_nodes, self.num_neighbors, self.local_radius
        padded = F.pad(demands.reshape(batch * steps, 1, self.height, self.width), (radius,) * 4)
        patches = F.unfold(padded, kernel_size=self.window_size)
        local_crop = patches.transpose(1, 2).reshape(batch, steps, nodes, neighbors)
        value_tokens = self.scalar_embedding(torch.log1p(torch.clamp(local_crop, min=0.0)).unsqueeze(-1))
        edge_token = self.special_embedding.weight[1].view(1, 1, 1, 1, -1)
        value_tokens = torch.where(self.neighbor_valid.view(1, 1, nodes, neighbors, 1), value_tokens, edge_token)
        if not self.use_neighbors:
            keep = torch.zeros(neighbors, dtype=value_tokens.dtype, device=value_tokens.device)
            keep[neighbors // 2] = 1.0
            value_tokens = value_tokens * keep.view(1, 1, 1, neighbors, 1)
        cls_token = self.special_embedding.weight[0].view(1, 1, 1, 1, -1) + self.node_embedding.view(1, 1, nodes, 1, -1)
        tokens = torch.cat([cls_token.expand(batch, steps, -1, -1, -1), value_tokens], dim=3)
        hidden = (tokens + self.position_embedding.view(1, 1, 1, 1 + neighbors, -1)).reshape(
            batch * steps * nodes, 1 + neighbors, -1
        )
        for layer in self.transformer:
            hidden = layer(hidden)
        return hidden[:, 0].reshape(batch, steps, nodes, -1)


__all__ = ["LocalHistoryEncoder"]
