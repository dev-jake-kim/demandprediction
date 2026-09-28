"""another_model-style local demand history encoder."""

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

        # persistent=True로 저장해야 한다: transformers 5.0의 from_pretrained는 모델을 meta
        # device에서 만든 뒤 체크포인트에 있는 키만 실체화하므로, persistent=False 버퍼는
        # 값이 복원되지 않고 깨진다(models/modeling.py의 idx_table/mask_table과 같은 이유).
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
        # SDPA 한계에 영향을 주는 어텐션 가중치 dropout만 별도로 제어한다.
        # 출력/FFN dropout은 기존 dropout 값 그대로 유지한다.
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

        # --- 노드별 weight offset (lora 브랜치 node_adaptive와 같은 구조) ---
        # self.history_lstm은 모듈 그대로 두고(체크포인트 키 호환 + PyTorch 기본 초기화 유지)
        # 여기의 delta만 더해서 쓴다. delta는 0으로 시작한다 — 학습 시작 시점의 모델이
        # node_adaptive를 끈 것과 완전히 동일해야 "여기에 노드별 offset만 추가"라는 전제가 선다.
        #
        # 전체 노드가 아니라 train 구간 평균 수요가 임계값을 넘는 노드에만 준다. 선택되지 않은
        # 노드는 아래 forward에서 nn.LSTM(cuDNN fused) 결과를 그대로 쓰므로 속도 손해가 없다.
        self.node_adaptive = node_adaptive_indices is not None
        if self.node_adaptive:
            # 값 검사는 파이썬 리스트에서 한다. from_pretrained가 meta device에서 __init__을
            # 돌기 때문에, 텐서로 만든 뒤 검사하면 meta 텐서에 .item()을 부르게 되어 깨진다
            # (modeling.py의 weather_mean/std와 같은 이유).
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
            # **버퍼로 만들면 안 된다.** transformers 5.0의 from_pretrained는 meta device에서
            # 모델을 만든 뒤 체크포인트에 있는 키만 실체화한다. 2-stage 학습의 stage 2는
            # node_adaptive를 끄고 학습한 stage 1 체크포인트에서 이어받는데, 거기엔 이 키가
            # 없으므로 버퍼가 torch.empty(=쓰레기값)로 남아 엉뚱한 노드를 고르게 된다
            # (실제로 확인함: [140688794353328, 93890308059328, ...]).
            # 노드 목록은 config에만 두고, 텐서는 forward에서 device별로 만들어 캐시한다.
            self._node_adaptive_index_list = index_list
            self._cached_node_index: Tensor | None = None
            self.node_delta_weight_ih = nn.Parameter(
                torch.zeros(num_adaptive, n_gates, d_model + extra_dim)
            )
            self.node_delta_weight_hh = nn.Parameter(
                torch.zeros(num_adaptive, n_gates, history_hidden)
            )
            # PyTorch LSTM은 bias_ih/bias_hh 두 개를 갖지만 계산에는 둘의 합만 들어간다
            # (cuDNN 호환용 중복). 노드별로 둘을 따로 두면 파라미터만 2배가 되고 표현력은
            # 그대로이므로 합에 더하는 하나만 둔다.
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
        """선택된 노드들의 마지막 시점 hidden을 노드별 W+ΔW로 직접 계산한다.

        ``sequence``: ``[A, k, d_model+extra_dim]`` (A = 선택된 노드 수 × 배치).
        실제로는 ``[B, A, k, F]``를 받아 ``[B, A, history_hidden]``을 돌려준다.

        nn.LSTM은 샘플(노드)별 가중치를 받을 수 없어서 셀을 직접 돈다. 가중치는
        ``self.history_lstm``이 그대로 들고 있고 여기서는 노드별 delta만 더해 쓴다.
        delta가 전부 0이면 이 경로의 출력은 nn.LSTM과 수치적으로 동일하다 — 그래야 측정된
        차이가 delta 때문임이 분리된다(``validate_merged.py``가 이걸 대조로 고정한다).
        """

        batch, num_adaptive, steps, _ = sequence.shape
        weight_ih = self.history_lstm.weight_ih_l0 + self.node_delta_weight_ih  # (A,4h,F)
        weight_hh = self.history_lstm.weight_hh_l0 + self.node_delta_weight_hh  # (A,4h,h)
        bias = (
            self.history_lstm.bias_ih_l0 + self.history_lstm.bias_hh_l0 + self.node_delta_bias
        )  # (A,4h)

        # 입력 투영은 h에 의존하지 않으므로 k개 시점을 한 번에 계산해 순차 구간을 절반으로 줄인다.
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
        # 원본은 numpy 이중 루프였다. 값은 동일하되 torch 팩토리만 쓰도록 바꾼 이유는
        # from_pretrained가 meta device 컨텍스트에서 __init__을 도는데, torch.from_numpy는
        # 그 컨텍스트를 무시하고 CPU 텐서를 만들어 device가 섞이기 때문이다.
        # 열 순서는 원본과 같다: column c -> dy = c // window - r, dx = c % window - r.
        radius, window = self.local_radius, self.window_size
        offsets = torch.arange(-radius, radius + 1)
        dy = offsets.repeat_interleave(window)  # [neighbors]
        dx = offsets.repeat(window)  # [neighbors]
        node = torch.arange(self.num_nodes)
        y = torch.div(node, self.width, rounding_mode="floor").unsqueeze(1) + dy.unsqueeze(0)
        x = (node % self.width).unsqueeze(1) + dx.unsqueeze(0)
        return (y >= 0) & (y < self.height) & (x >= 0) & (x < self.width)

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
        node_hidden = hidden[-1].reshape(batch, nodes, -1)

        if self.node_adaptive:
            # 선택된 노드만 노드별 W+ΔW로 다시 계산해 덮어쓴다. 나머지 노드는 위의 cuDNN
            # fused LSTM 결과를 그대로 쓰므로 node_adaptive를 껐을 때와 완전히 동일하다.
            # (선택 노드도 cuDNN으로 한 번 계산되지만 그쪽은 fused라 비용이 거의 없다.)
            index = self.node_adaptive_index(sequence.device)
            adaptive_sequence = sequence.reshape(batch, nodes, steps, -1).index_select(1, index)
            adaptive_hidden = self._node_adaptive_hidden(adaptive_sequence)
            # in-place 대입 대신 out-of-place index_copy를 쓴다 — autograd가 두 경로의
            # gradient를 모두 받아야 하고, 학습 중 in-place 수정은 backward를 깨뜨린다.
            node_hidden = node_hidden.index_copy(1, index, adaptive_hidden)

        return local_crop, node_hidden


__all__ = ["LocalHistoryEncoder"]
