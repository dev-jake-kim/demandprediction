"""재설계된 수요 예측 모델 — 골격.

**지금 학습 가능한 것은 local 관점 하나짜리 모델이다.** ``ContextEncoder``,
:class:`LocalViewEncoder`, :class:`PredictionHead`, 그리고 ``MergedDemandModel.forward``가
구현돼 있어 end-to-end로 돈다. 나머지 세 관점(:class:`PeriodicViewEncoder`,
:class:`RetrievalViewEncoder`)과 :class:`ViewFusion`은 ``forward``가 ``pass``이고,
파라미터가 하나도 없어서 모델에 매달려 있어도 학습에 영향을 주지 않는다. 비어 있는
블록이 확정하는 것은 **정보의 흐름**, 즉 어떤 정보를 받아 어떤 shape로 내보내는지뿐이다.

거시적 구조::

    demand_history [B,k,H,W] ─┬─> LocalViewEncoder      ──> [B,N,D]  "내 주변에서 최근 k시간"
                              │
                              └─> RetrievalViewEncoder  ──> [B,N,D]  "과거의 비슷했던 n개 시점" (미구현)
    daily_demand   [B,Ld,N,1] ──> PeriodicViewEncoder   ──> [B,N,D]  "며칠 전 같은 시간대"     (미구현)
    weekly_demand  [B,Lw,N,1] ──> PeriodicViewEncoder   ──> [B,N,D]  "몇 주 전 같은 요일·시간대" (미구현)

                              stack -> [B,N,4,D]
                                         │
                                    ViewFusion   -> [B,N,D]
                                         │
                                  PredictionHead -> [B,N] -> reshape -> [B,H,W]

네 관점이 전부 **노드당 D차원 벡터 하나**라는 같은 규약으로 끝나는 것이 이 설계의 전부다.
관점을 더하거나 빼는 일이 stack의 항목을 더하거나 빼는 일이 된다.

축 이름 규약 (파일 전체에서 동일):

===== ==========================================================================
``B``  배치
``k``  최근 히스토리 길이 (``config.time_step``, 기본 24시간)
``H``  격자 세로, ``W`` 격자 가로, ``N = H*W`` 노드 수
``Ld`` 일 주기 lag 개수, ``Lw`` 주 주기 lag 개수 (dataset이 정한다)
``n``  검색해 오는 유사 시점 개수 (``config.num_retrieval``)
``C``  보조 정보 차원 (``config.context_dim`` = 날씨 3 + 요일 + 시간대)
``D``  노드 임베딩 차원 (``config.d_model``)
===== ==========================================================================
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from .config import MergedDemandConfig
from .losses import SCALAR_LOSS_TYPES, build_loss


class ContextEncoder(nn.Module):
    """시점별 보조 정보(날씨 + 캘린더)를 한 벡터로 묶는다.

    네 관점이 전부 이걸 공유한다. 관점마다 시점 축의 의미가 다를 뿐(최근 k시간 / 일 주기 lag /
    주 주기 lag) 묶는 방식은 같아서, 여기 한 곳에서만 정의한다.

    날씨는 임베딩하지 않고 train 구간 기준 min-max로 정규화한 값을 그대로 쓴다. 두 채널
    모두 train 구간에서 [0, 1]에 들어가고, val/test가 그 범위를 넘으면 1을 넘을 수 있다
    (clip하지 않는다 — 실제로 더 더웠거나 더 많이 온 것이므로 그 정보를 지울 이유가 없다).

    **CSV의 3채널을 2채널로 합친다**: 강수량(mm)과 적설(cm)은 같은 현상(하늘에서 떨어지는
    물)을 단위만 달리 잰 것이라, 학습 가능한 환산계수 하나로 묶는다::

        총강수 = 강수량 + snow_scale * 적설

    ``snow_scale``의 초깃값 1.0은 "적설 1cm ≈ 강수 1mm"라는 관례적 환산이고, 그 비율이
    수요에 미치는 영향은 데이터가 정하도록 학습시킨다.

    정규화 분모(``config.precipitation_max``)는 ``snow_scale``이 학습 중 계속 변하기 때문에
    데이터에서 그때그때 다시 잴 수 없다. **snow_scale=1.0 기준으로 train 구간에서 한 번 재서
    하이퍼파라미터로 고정**한다. 분모에 학습 파라미터가 없어야 gradient도 깨끗하다.

    이렇게 묶는 실질적 이유는 적설을 따로 정규화할 수 없기 때문이다. ulsan train 구간에서
    적설이 0이 아닌 시점은 0.31%뿐이라 std가 0.0115까지 떨어지고, 그러면 눈이 온 시점의
    정규화 값이 26σ까지 튀어 나머지 입력을 압도한다. porto는 적설이 train 구간 내내 정확히
    0이라 std가 0이고(train.py가 1e-6으로 clip한다) 채널 전체가 상수 0인 죽은 입력이었다.
    강수량에 합치면 두 문제가 같이 사라진다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        missing = [
            name
            for name in ('temperature_min', 'temperature_max', 'precipitation_max')
            if getattr(config, name) is None
        ]
        if missing:
            raise ValueError(
                f'날씨 정규화 기준이 없음: {missing}. train 구간에서 재서 '
                'configs/model/merged_<city>.yaml에 적어야 한다 — val/test를 보면 시간 리크다'
            )
        # 값은 config에만 둔다(버퍼로 만들면 from_pretrained의 meta device 초기화에서
        # 복원 여부를 따로 챙겨야 한다). 스칼라라 텐서와 그냥 연산된다.
        self.temperature_min = float(config.temperature_min)
        self.temperature_range = float(config.temperature_max) - self.temperature_min
        self.precipitation_max = float(config.precipitation_max)
        if self.temperature_range <= 0:
            raise ValueError('temperature_max는 temperature_min보다 커야 함')
        if self.precipitation_max <= 0:
            raise ValueError('precipitation_max는 양수여야 함')
        # 적설 -> 강수량 환산계수. 1.0(적설 1cm = 강수 1mm)에서 출발해 학습으로 조정된다.
        # 스칼라 하나라 모든 시점/노드가 같은 값을 쓴다.
        self.snow_scale = nn.Parameter(torch.tensor(1.0))
        # 요일/시간대만 임베딩한다. 세 관점이 같은 테이블을 공유한다 — 요일 3은 어느
        # 관점에서나 요일 3이다.
        self.weekday_embedding = nn.Embedding(7, config.weekday_dim)
        self.hour_embedding = nn.Embedding(24, config.hour_dim)

    def forward(self, weather: Tensor, hour_of_day: Tensor, day_of_week: Tensor) -> Tensor:
        """
        Args:
            weather: ``[B, L, 3]`` 원본 스케일 (기온, 강수량, 적설).
            hour_of_day: ``[B, L]`` 0..23 정수.
            day_of_week: ``[B, L]`` 0..6 정수.

        Returns:
            ``[B, L, C]`` 시점별 보조 정보.
        """

        temperature, rainfall, snowfall = weather.unbind(dim=-1)

        # 정규화를 여기 한 곳에서만 한다 — 날것의 기온(수십 단위)이 log1p 수요를 압도하는
        # 일이 없어야 한다.
        temperature_norm = (temperature - self.temperature_min) / self.temperature_range

        # 강수는 최소값이 0이라 (x - min)/(max - min)이 x/max로 줄어든다. 덕분에 "비도 눈도
        # 안 왔다"가 정확히 0이 된다 — 정보 없음과 같은 값이다.
        precipitation = rainfall + self.snow_scale * snowfall
        precipitation_norm = precipitation / self.precipitation_max

        return torch.cat(
            [
                torch.stack([temperature_norm, precipitation_norm], dim=-1),
                self.weekday_embedding(day_of_week),
                self.hour_embedding(hour_of_day),
            ],
            dim=-1,
        )


def crop_local_windows(demands: Tensor, radius: int) -> Tensor:
    """``[B, k, H, W]`` -> ``[B, k, N, P]``. 노드마다 자기 주변 (2a+1)^2 창을 떼어낸다.

    격자 밖은 0으로 패딩된다 — "수요가 0"과 구분되지 않으므로, 어디가 격자 밖인지는
    :func:`make_neighbor_valid`가 따로 알려준다.

    ``F.unfold``는 행 우선이라 P축의 순서는 (dy, dx)가 -a부터 +a까지 도는 순서이고,
    따라서 **중앙(자기 노드)은 항상 ``P // 2``번**이다.
    """

    batch, steps, height, width = demands.shape
    window = 2 * radius + 1
    padded = F.pad(demands.reshape(batch * steps, 1, height, width), (radius,) * 4)
    patches = F.unfold(padded, kernel_size=window)  # [B*k, P, N]
    return patches.transpose(1, 2).reshape(batch, steps, height * width, window * window)


def make_neighbor_valid(height: int, width: int, radius: int) -> Tensor:
    """``[N, P]`` bool. 노드 n의 P번째 이웃이 격자 안에 실제로 존재하는가.

    :func:`crop_local_windows`의 P축 순서와 정확히 같은 규약으로 만든다.
    numpy가 아니라 torch 팩토리만 쓰는 이유는 ``from_pretrained``가 meta device
    컨텍스트에서 ``__init__``을 돌기 때문이다(numpy 경유는 그 컨텍스트를 무시한다).
    """

    window = 2 * radius + 1
    offsets = torch.arange(-radius, radius + 1)
    dy = offsets.repeat_interleave(window)  # [P]
    dx = offsets.repeat(window)  # [P]
    node = torch.arange(height * width)
    y = torch.div(node, width, rounding_mode='floor').unsqueeze(1) + dy.unsqueeze(0)
    x = (node % width).unsqueeze(1) + dx.unsqueeze(0)
    return (y >= 0) & (y < height) & (x >= 0) & (x < width)


class FourierScalarEmbedding(nn.Module):
    """스칼라 하나 -> ``D``차원 토큰. 학습 가능한 Fourier 특징 + 선형 투영.

    수요는 0, 1, 2 같은 작은 정수가 대부분이라 선형 투영만으로는 인접한 값이 거의
    구분되지 않는다. 주파수를 여러 개 걸어 그 차이를 벌린다.
    """

    def __init__(self, d_model: int, num_bands: int) -> None:
        super().__init__()
        self.log_frequencies = nn.Parameter(torch.linspace(-2.0, 2.0, num_bands))
        self.projection = nn.Linear(1 + 2 * num_bands, d_model)

    def forward(self, value: Tensor) -> Tensor:
        """``[..., 1]`` -> ``[..., D]``."""

        frequencies = self.log_frequencies.exp().view(*([1] * (value.ndim - 1)), -1)
        phase = value * frequencies * (2.0 * math.pi)
        return self.projection(torch.cat([value, phase.sin(), phase.cos()], dim=-1))


class LocalViewEncoder(nn.Module):
    """관점 1 — 공간. "내 주변 (2a+1)^2 칸에서 최근 k시간 동안 무슨 일이 있었나."

    유일하게 이웃 노드의 수요를 보는 관점이다. 나머지 세 관점은 노드별로 독립이다.

    두 단계로 접는다::

        [B,k,H,W] --crop-->  [B,k,N,P]      노드별 창
                  --embed--> [B,k,N,1+P,D]  창 안의 각 칸이 토큰 하나, 맨 앞은 CLS
                  --transformer + CLS -->   [B,k,N,D]   (공간 축을 접음)
                  --LSTM -->                [B,N,D]     (시간 축을 접음)

    Transformer는 **창 하나**를 본다 — (배치, 시점, 노드)를 전부 batch 축으로 접으므로
    ``B*k*N``개의 길이 ``1+P`` 시퀀스가 한 번에 들어간다. ulsan(B=8,k=24,N=168) 기준
    32,256개다. 창 안의 공간 배치는 attention이 직접 알 수 없고 ``position_embedding``이
    학습으로 담는다.

    ``config.zero_node_indices``로 학습에서 뺀 노드가 있으면 **crop 직후 잘라낸다**. 이후
    모든 단계(토큰 임베딩/Transformer/LSTM)가 노드별로 독립이라 남은 노드의 결과는 전부
    계산한 뒤 버리는 것과 수치적으로 같고, 시퀀스 수만 줄어 그만큼 빨라진다. 뺀 노드도
    *이웃으로서는* 그대로 남는다 — crop이 이미 끝난 뒤에 자르므로 남은 노드의 창 안에
    입력값으로 계속 들어간다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        if config.d_model % config.transformer_heads != 0:
            raise ValueError('d_model은 transformer_heads로 나누어떨어져야 함')
        if config.node_adaptive and (
            config.zero_node_max_demand is not None or config.zero_node_indices
        ):
            raise ValueError(
                'node_adaptive=true는 zero_node_max_demand/zero_node_indices와 '
                '동시에 사용할 수 없음: 적응 노드 id와 잘린 노드 축이 어긋남'
            )

        d_model = config.d_model
        self.local_radius = config.local_radius
        self.num_nodes = config.num_nodes
        self.num_neighbors = config.num_neighbors

        # 결과를 내야 하는 노드 id. None이면 전부. 학습에서 뺀 노드를 crop 직후 잘라내는 데 쓴다.
        # 텐서가 아니라 파이썬 리스트로 들고 있는다 — from_pretrained가 meta device에서
        # __init__을 돌기 때문에 버퍼로 만들면 체크포인트에 없을 때 쓰레기값이 남는다.
        self.active_node_ids: list[int] | None = None
        if config.zero_node_indices:
            excluded = set(int(v) for v in config.zero_node_indices)
            self.active_node_ids = [n for n in range(self.num_nodes) if n not in excluded]
        self.num_active_nodes = (
            self.num_nodes if self.active_node_ids is None else len(self.active_node_ids)
        )
        self._cached_active_index: Tensor | None = None

        # persistent=True여야 한다: from_pretrained는 meta device에서 모델을 만든 뒤
        # 체크포인트에 있는 키만 실체화하므로, persistent=False 버퍼는 값이 복원되지 않는다.
        self.register_buffer(
            'neighbor_valid',
            make_neighbor_valid(config.height, config.width, config.local_radius),
            persistent=True,
        )
        self.scalar_embedding = FourierScalarEmbedding(d_model, config.num_fourier_bands)
        # 0 = CLS(요약을 모으는 자리), 1 = EDGE(격자 밖 = 값이 없음).
        self.special_embedding = nn.Embedding(2, d_model)
        # "몇 번 노드의 요약인지"를 CLS에 새겨 넣는다.
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        # 창 안의 자리(CLS 1개 + 이웃 P개)를 구분하는 위치 임베딩.
        self.position_embedding = nn.Parameter(
            torch.randn(1 + self.num_neighbors, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        # nn.TransformerEncoderLayer는 dropout 인자 하나를 네 곳에 다 쓴다. 그중 어텐션
        # 가중치에 걸리는 것만 따로 떼어 다시 설정한다 — TransformerEncoder가 이 레이어를
        # deepcopy하므로 복제 전에 바꿔야 모든 층에 반영된다.
        encoder_layer.self_attn.dropout = config.attention_dropout
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        # 시간 축을 접는다. 입력은 시점별 창 요약(D) + 그 시점의 보조 정보(C).
        self.temporal_lstm = nn.LSTM(d_model + config.context_dim, d_model, batch_first=True)
        self.shared_weight_fp8 = bool(config.shared_weight_fp8)
        if self.shared_weight_fp8 and not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError(
                'shared_weight_fp8=true에는 torch.float8_e4m3fn을 지원하는 PyTorch가 필요함'
            )

        # 적응 노드 id는 config의 파이썬 리스트로만 보관한다. 버퍼로 등록하면
        # from_pretrained의 meta 초기화에서 체크포인트에 없는 id가 torch.empty 쓰레기값이 된다.
        self.node_adaptive = bool(config.node_adaptive)
        self._node_adaptive_index_list: list[int] | None = None
        self._cached_node_adaptive_index: Tensor | None = None
        if self.node_adaptive:
            if config.node_adaptive_indices is None:
                raise ValueError(
                    'node_adaptive=true인데 node_adaptive_indices가 주어지지 않음'
                )
            index_list = [int(value) for value in config.node_adaptive_indices]
            if not index_list:
                raise ValueError('node_adaptive_indices는 비어 있지 않은 목록이어야 함')
            if min(index_list) < 0 or max(index_list) >= config.num_nodes:
                raise ValueError(
                    f'node_adaptive_indices가 노드 범위(0..{config.num_nodes - 1})를 벗어남'
                )
            if len(set(index_list)) != len(index_list):
                raise ValueError('node_adaptive_indices에 중복이 있음')
            self._node_adaptive_index_list = index_list

        # s와 ΔW는 fp32 학습 파라미터다. s=1, ΔW=0이면 fp8 off에서 baseline
        # temporal_lstm과 같은 함수가 되고, fp8 on에서는 공유 weight만 양자화된다.
        if self.node_adaptive:
            num_adaptive = len(self._node_adaptive_index_list)
            n_gates = 4 * d_model
            self.s = nn.Parameter(torch.tensor(1.0))
            self.node_delta_weight_ih = nn.Parameter(
                torch.zeros(num_adaptive, n_gates, d_model + config.context_dim)
            )
            self.node_delta_weight_hh = nn.Parameter(
                torch.zeros(num_adaptive, n_gates, d_model)
            )
            self.node_delta_bias = nn.Parameter(torch.zeros(num_adaptive, n_gates))

    def node_adaptive_index(self, device: torch.device) -> Tensor:
        """ΔW를 받는 전역 노드 id 텐서. config 리스트를 device별로 캐시한다."""

        if not self.node_adaptive or self._node_adaptive_index_list is None:
            raise RuntimeError('node_adaptive가 꺼져 있어 adaptive index가 없음')
        cached = self._cached_node_adaptive_index
        if cached is None or cached.device != device:
            cached = torch.as_tensor(
                self._node_adaptive_index_list, dtype=torch.long, device=device
            )
            self._cached_node_adaptive_index = cached
        return cached

    @staticmethod
    def _fake_quantize_fp8(weight: Tensor) -> tuple[Tensor, Tensor]:
        """absmax scaled FP8 fake quantization with a straight-through gradient.

        Returns the fp32 STE value used by both functional and manual LSTM paths and
        the fp32 scale used to map the tensor to the e4m3fn representable range.
        """

        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError('설치된 torch에 torch.float8_e4m3fn이 없음')
        absmax = weight.detach().abs().amax()
        # FP8 e4m3fn의 최대 유한값 448을 활용한다. all-zero tensor도 안전하게
        # 양자화할 수 있도록 scale=1을 쓴다.
        scale = torch.where(
            absmax > 0,
            absmax / weight.new_tensor(448.0),
            weight.new_tensor(1.0),
        )
        quantized = (weight / scale).to(torch.float8_e4m3fn)
        dequantized = quantized.to(weight.dtype) * scale
        ste = weight + (dequantized - weight).detach()
        return ste, scale

    def shared_lstm_weights(self, *, quantize: bool | None = None) -> tuple[Tensor, ...]:
        """Return the four shared temporal-LSTM tensors used by both execution paths.

        ``quantize`` defaults to the config switch and exists to make CPU acceptance
        checks able to compare the exact functional fp32 and fake-FP8 paths.
        """

        if quantize is None:
            quantize = self.shared_weight_fp8
        params = (
            self.temporal_lstm.weight_ih_l0,
            self.temporal_lstm.weight_hh_l0,
            self.temporal_lstm.bias_ih_l0,
            self.temporal_lstm.bias_hh_l0,
        )
        if not quantize:
            return params
        if not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError('설치된 torch에 torch.float8_e4m3fn이 없음')
        return tuple(self._fake_quantize_fp8(param)[0] for param in params)

    def functional_lstm(
        self, sequence: Tensor, weights: tuple[Tensor, ...] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run the temporal LSTM with externally supplied weights.

        ``torch._VF.lstm`` takes ``(input, hx, params, has_biases, num_layers,
        dropout, train, bidirectional, batch_first)``. Keeping this call in one
        method ensures the non-adaptive path uses exactly the same (possibly FP8
        fake-quantized) tensors as the adaptive cell loop.
        """

        if weights is None:
            weights = self.shared_lstm_weights()
        hidden0 = sequence.new_zeros(1, sequence.shape[0], self.temporal_lstm.hidden_size)
        cell0 = sequence.new_zeros(1, sequence.shape[0], self.temporal_lstm.hidden_size)
        output, hidden, cell = torch._VF.lstm(
            sequence,
            (hidden0, cell0),
            list(weights),
            True,
            1,
            0.0,
            self.training,
            False,
            True,
        )
        return output, (hidden, cell)

    def _node_adaptive_hidden(
        self, sequence: Tensor, shared_weights: tuple[Tensor, ...]
    ) -> Tensor:
        """Compute adaptive-node hidden states with the gated convex combination."""

        batch, num_adaptive, steps, _ = sequence.shape
        weight_ih_shared, weight_hh_shared, bias_ih_shared, bias_hh_shared = shared_weights
        one_minus_s = 1.0 - self.s
        weight_ih = (
            self.s * weight_ih_shared.unsqueeze(0)
            + one_minus_s * self.node_delta_weight_ih
        )
        weight_hh = (
            self.s * weight_hh_shared.unsqueeze(0)
            + one_minus_s * self.node_delta_weight_hh
        )
        bias = (
            self.s * (bias_ih_shared + bias_hh_shared).unsqueeze(0)
            + one_minus_s * self.node_delta_bias
        )

        # Input projection is independent of h, so project all k steps once.
        gates_x = torch.einsum('bakf,agf->bakg', sequence, weight_ih) + bias.unsqueeze(1)
        hidden = sequence.new_zeros(batch, num_adaptive, self.temporal_lstm.hidden_size)
        cell = sequence.new_zeros(batch, num_adaptive, self.temporal_lstm.hidden_size)
        for step in range(steps):
            gates = gates_x[:, :, step] + torch.einsum(
                'bah,agh->bag', hidden, weight_hh
            )
            in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=-1)
            # PyTorch LSTM gate order is i, f, g, o.
            cell = forget_gate.sigmoid() * cell + in_gate.sigmoid() * cell_gate.tanh()
            hidden = out_gate.sigmoid() * cell.tanh()
        return hidden

    def active_node_index(self, device: torch.device) -> Tensor | None:
        """결과를 내는 노드 id 텐서. 전부 계산하면 None. device별로 캐시한다."""

        if self.active_node_ids is None:
            return None
        cached = self._cached_active_index
        if cached is None or cached.device != device:
            cached = torch.as_tensor(self.active_node_ids, dtype=torch.long, device=device)
            self._cached_active_index = cached
        return cached

    def forward(self, demand_history: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            demand_history: ``[B, k, H, W]`` 최근 k시간의 격자 전체 수요(raw 스케일).
            context: ``[B, k, C]`` 같은 k시점의 보조 정보.

        Returns:
            ``[B, N_active, D]`` 노드별 공간 관점 임베딩. 학습에서 뺀 노드가 없으면
            ``N_active == N``이고, 있으면 남은 노드만 ``active_node_ids`` 순서로 돌려준다.
        """

        local_crop = crop_local_windows(demand_history, self.local_radius)
        # 여기서 자른다 — crop이 끝난 뒤라 뺀 노드도 남은 노드의 창 안에는 그대로 들어 있다.
        active = self.active_node_index(local_crop.device)
        neighbor_valid = self.neighbor_valid
        if active is not None:
            local_crop = local_crop.index_select(2, active)
            neighbor_valid = neighbor_valid.index_select(0, active)
        batch, steps, nodes, neighbors = local_crop.shape

        # 수요는 꼬리가 긴 분포라 log1p로 압축한 뒤 토큰으로 만든다.
        values = torch.log1p(torch.clamp(local_crop, min=0.0)).unsqueeze(-1)
        tokens = self.scalar_embedding(values)  # [B,k,N,P,D]
        # 격자 밖은 "수요 0"이 아니라 "값이 없음"이다 — 전용 EDGE 토큰으로 갈아끼운다.
        edge_token = self.special_embedding.weight[1].view(1, 1, 1, 1, -1)
        tokens = torch.where(
            neighbor_valid.view(1, 1, nodes, neighbors, 1), tokens, edge_token
        )

        node_embedding = self.node_embedding
        if active is not None:
            node_embedding = node_embedding.index_select(0, active)
        cls_token = self.special_embedding.weight[0].view(1, 1, 1, 1, -1)
        cls_token = cls_token + node_embedding.view(1, 1, nodes, 1, -1)
        cls_token = cls_token.expand(batch, steps, -1, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=3)
        tokens = tokens + self.position_embedding.view(1, 1, 1, 1 + neighbors, -1)

        # (배치, 시점, 노드)를 batch 축으로 접는다 — 창 하나가 시퀀스 하나다.
        encoded = self.transformer(tokens.reshape(batch * steps * nodes, 1 + neighbors, -1))
        summary = encoded[:, 0].reshape(batch, steps, nodes, -1)  # CLS만 꺼낸다

        # 시간 축을 접는다. 보조 정보는 노드에 무관하므로 노드 축으로 브로드캐스트해 붙인다.
        sequence = summary.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        expanded_context = (
            context[:, None]
            .expand(batch, nodes, steps, context.shape[-1])
            .reshape(batch * nodes, steps, -1)
        )
        sequence = torch.cat([sequence, expanded_context], dim=-1)
        shared_weights = self.shared_lstm_weights()
        _, (hidden, _) = self.functional_lstm(sequence, shared_weights)
        node_hidden = hidden[-1].reshape(batch, nodes, -1)
        if self.node_adaptive:
            # Non-adaptive nodes stay on the external-weight functional LSTM above.
            # Adaptive nodes are recomputed with the same shared (possibly FP8)
            # tensors and their gated convex combination, then merged out-of-place.
            index = self.node_adaptive_index(sequence.device)
            adaptive_sequence = sequence.reshape(batch, nodes, steps, -1).index_select(1, index)
            adaptive_hidden = self._node_adaptive_hidden(adaptive_sequence, shared_weights)
            node_hidden = node_hidden.index_copy(1, index, adaptive_hidden)
        return node_hidden


class PeriodicViewEncoder(nn.Module):
    """관점 2·3 — 주기. "같은 시간대의 과거 값들이 어떻게 움직였나."

    daily(하루 주기)와 weekly(일주일 주기)가 이 클래스의 서로 다른 인스턴스다. 둘의 차이는
    dataset이 뽑아 주는 lag 집합뿐이라 코드를 나누지 않는다.

    이 관점은 노드별로 독립이다 — 이웃 노드를 보지 않는다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()

    def forward(self, lag_demand: Tensor, lag_mask: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            lag_demand: ``[B, L, N, 1]`` 각 lag 시점의 노드별 수요(raw 스케일).
                ``L``은 daily면 ``Ld``, weekly면 ``Lw``.
            lag_mask: ``[B, L]`` bool. **True가 무효**(그 lag이 데이터 시작 이전이라 없음).
                무효 시점은 ``lag_demand``에도 0이 채워져 있다.
            context: ``[B, L, C]`` 각 lag 시점의 보조 정보.

        Returns:
            ``[B, N, D]`` 노드별 주기 관점 임베딩.
        """

        pass


class CausalRetrieval(nn.Module):
    """관점 4 — 유추. "지금과 비슷했던 과거 n개 시점의 **다음 시간**에는 얼마였나."

    ``tmp`` 브랜치 ``models/merged/modules/retrieval.py``를 그대로 옮긴 것이다. 계산은
    바뀌지 않았고 config 필드 이름만 현재 스키마(``num_retrieval``)에 맞췄다.

    - 질의는 노드별 로컬 윈도우 crop ``[k, P]``를 편 ``k*P``차원 벡터다(이웃까지 본다).
    - 후보는 같은 노드의 과거 윈도우이고 인과 경계는 ``tau < target_time``이다
      (``retrieval_scope='train_prefix'``면 train split 끝까지 더 자른다).
    - top-k 코사인 유사도를 softmax로 정규화해 ``grid[tau]``(윈도우 직후 실제 수요)를
      가중 평균한다. 출력은 raw 스케일 스칼라 ``[B, N]``이다.

    학습 파라미터가 없고 gradient도 흐르지 않는다. 결과는 ``target_time``에만 의존하므로
    ``[T, N]`` 캐시에 저장해 두 번째 epoch부터는 계산을 건너뛴다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        if config.retrieval_scope not in ('observed_past', 'train_prefix'):
            raise ValueError("retrieval_scope는 'observed_past' | 'train_prefix'여야 함")
        if config.num_retrieval <= 0 or config.retrieval_chunk_size <= 0:
            raise ValueError('num_retrieval / retrieval_chunk_size는 1 이상이어야 함')

        self.height = config.height
        self.width = config.width
        self.num_nodes = config.num_nodes
        self.time_step = config.time_step
        self.local_radius = config.local_radius
        self.window_size = config.window_size
        self.num_neighbors = config.num_neighbors
        self.flat_query_dim = config.time_step * self.num_neighbors
        self.retrieval_k = config.num_retrieval
        self.retrieval_chunk_size = config.retrieval_chunk_size
        self.retrieval_scope = config.retrieval_scope
        self.retrieval_train_end = config.retrieval_train_end
        if self.retrieval_scope == 'train_prefix' and self.retrieval_train_end is None:
            raise ValueError("retrieval_scope='train_prefix'에는 retrieval_train_end가 필요함")
        self._grid: Tensor | None = None
        self._crops: Tensor | None = None
        self._cache: Tensor | None = None
        if config.retrieval_grid_path is not None:
            self._load_grid(config.retrieval_grid_path)

    def _load_grid(self, grid_path: str | Path) -> None:
        path = Path(grid_path).expanduser().resolve()
        grid = np.load(path, mmap_mode='r')
        if grid.ndim != 3 or tuple(grid.shape[1:]) != (self.height, self.width):
            raise ValueError(
                f'검색 대상 grid는 [T,{self.height},{self.width}]여야 함 '
                f'(받음: {tuple(grid.shape)})'
            )
        if not np.isfinite(grid).all() or np.min(grid) < 0:
            raise ValueError('검색 대상 grid는 유한하고 음이 아닌 값이어야 함')
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32, copy=True))
        self._grid = grid_tensor
        padded = F.pad(grid_tensor.unsqueeze(1), (self.local_radius,) * 4)
        patches = F.unfold(padded, kernel_size=self.window_size)
        self._crops = patches.transpose(1, 2).contiguous()  # [T, N, P]
        # torch.full 대신 new_full: from_pretrained가 meta device 컨텍스트에서 __init__을
        # 도는데, 팩토리 함수는 그 컨텍스트를 따라가 meta 텐서가 되어버린다. grid_tensor는
        # from_numpy로 만든 실제 CPU 텐서라 여기서 device가 CPU로 고정된다.
        self._cache = grid_tensor.new_full(
            (grid_tensor.shape[0], self.num_nodes), float('nan')
        )
        self.grid_path = str(path)

    @torch.no_grad()
    def forward(self, local_crop: Tensor, sample_idx: Tensor) -> Tensor:
        """``[B, k, N, P]`` 질의 -> ``[B, N]`` 검색 예측값(raw 스케일)."""

        batch, steps, nodes, neighbors = local_crop.shape
        if self._grid is None or self._crops is None:
            return local_crop.new_zeros((batch, nodes))
        if steps != self.time_step or nodes != self.num_nodes or neighbors != self.num_neighbors:
            raise ValueError('검색 질의 shape가 설정과 다름')

        search_device = local_crop.device
        query = local_crop.detach().to(device=search_device, dtype=torch.float32)
        query = query.permute(0, 2, 3, 1).reshape(batch, nodes, self.flat_query_dim)
        query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sample_times = sample_idx.detach().to(device='cpu', dtype=torch.long).tolist()
        total_steps = self._grid.shape[0]
        result = torch.zeros(batch, nodes, dtype=torch.float32)
        uncached_positions: list[int] = []
        uncached_times: list[int] = []

        for position, target_time in enumerate(sample_times):
            target_time = int(target_time)
            if self._cache is not None and 0 <= target_time < self._cache.shape[0]:
                cached = self._cache[target_time]
                if torch.isfinite(cached).all():
                    result[position] = cached
                    continue
            uncached_positions.append(position)
            uncached_times.append(target_time)

        if not uncached_positions:
            return result.to(device=local_crop.device)

        start = self.time_step
        end_limits = []
        for target_time in uncached_times:
            end = target_time
            if self.retrieval_scope == 'train_prefix':
                end = min(end, int(self.retrieval_train_end))
            end_limits.append(min(end, total_steps))
        max_end = max(end_limits)
        if max_end <= start:
            for target_time in uncached_times:
                if self._cache is not None and 0 <= target_time < total_steps:
                    self._cache[target_time] = 0.0
            return result.to(device=local_crop.device)

        query_batch = query[uncached_positions]
        end_tensor = torch.tensor(end_limits, dtype=torch.long, device=search_device)
        best_scores: Tensor | None = None
        best_times: Tensor | None = None
        for chunk_start in range(start, max_end, self.retrieval_chunk_size):
            chunk_end = min(chunk_start + self.retrieval_chunk_size, max_end)
            candidate_times = torch.arange(
                chunk_start, chunk_end, dtype=torch.long, device=search_device
            )
            base = self._crops[chunk_start - self.time_step : chunk_end - 1]
            windows = base.unfold(0, self.time_step, 1)
            candidates = windows.permute(1, 0, 2, 3).reshape(nodes, -1, self.flat_query_dim)
            candidates = candidates.to(device=search_device)
            candidate_norm = candidates.norm(dim=-1).clamp_min(1e-8)
            similarity = torch.einsum('bnd,ncd->bnc', query_batch, candidates) / candidate_norm
            similarity = similarity.nan_to_num(0.0, 0.0, 0.0)
            valid_candidates = candidate_times.view(1, 1, -1) < end_tensor.view(-1, 1, 1)
            similarity = similarity.masked_fill(
                ~valid_candidates, torch.finfo(similarity.dtype).min
            )

            take = min(self.retrieval_k, similarity.shape[-1])
            chunk_scores, chunk_indices = similarity.topk(take, dim=-1)
            chunk_times = candidate_times[chunk_indices]
            if best_scores is None:
                best_scores, best_times = chunk_scores, chunk_times
            else:
                merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
                merged_times = torch.cat([best_times, chunk_times], dim=-1)
                take = min(self.retrieval_k, merged_scores.shape[-1])
                best_scores, order = merged_scores.topk(take, dim=-1)
                best_times = merged_times.gather(-1, order)

        assert best_scores is not None and best_times is not None
        has_candidates = end_tensor > start
        valid_best = best_times < end_tensor.view(-1, 1, 1)
        masked_scores = best_scores.masked_fill(~valid_best, torch.finfo(best_scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=-1).to(device='cpu')
        all_values = self._grid[torch.arange(start, max_end, dtype=torch.long)]
        all_values = all_values.reshape(-1, nodes).transpose(0, 1)
        indices = (best_times - start).clamp_min(0).to(device='cpu')
        values = all_values.unsqueeze(0).expand(len(uncached_positions), -1, -1).gather(2, indices)
        retrieved = (weights * values).sum(dim=-1)
        retrieved[~has_candidates] = 0.0

        for row, position, target_time in zip(retrieved, uncached_positions, uncached_times):
            result[position] = row
            if self._cache is not None and 0 <= target_time < total_steps:
                self._cache[target_time] = row
        return result.to(device=local_crop.device)


class NeuralRetrievalGate(nn.Module):
    """뉴럴 예측과 검색 예측을 **출력 단계에서** 섞는다.

    ``tmp`` 브랜치 ``modules/fusion.py``의 구조를 옮겼다. 검색이 임베딩이 아니라 raw
    스케일 스칼라를 내므로 융합도 임베딩 공간이 아니라 예측값 공간에서 일어난다::

        lambda = sigmoid(Linear([fused, retrieved]))
        prediction = lambda * neural_pred + (1 - lambda) * retrieved

    ``lambda_layer``를 0으로 초기화해 ``lambda=0.5``에서 출발한다 — 어느 쪽으로도 기울지
    않은 지점이고, 두 예측의 신뢰도는 학습이 정한다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        self.lambda_layer = nn.Linear(config.d_model + 1, 1)
        nn.init.zeros_(self.lambda_layer.weight)
        nn.init.zeros_(self.lambda_layer.bias)

    def forward(
        self, fused: Tensor, neural_pred: Tensor, retrieved: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            fused: ``[B, N, D]`` 융합된 노드 임베딩.
            neural_pred: ``[B, N]`` 뉴럴 예측(raw 스케일, >= 0).
            retrieved: ``[B, N]`` 검색 예측(raw 스케일, >= 0).

        Returns:
            ``lambda_weight``: ``[B, N]`` 뉴럴 쪽 신뢰도.
            ``prediction``: ``[B, N]`` 최종 예측.
        """

        gate_input = torch.cat([fused, retrieved.unsqueeze(-1)], dim=-1)
        lambda_weight = torch.sigmoid(self.lambda_layer(gate_input)).squeeze(-1)
        return lambda_weight, lambda_weight * neural_pred + (1.0 - lambda_weight) * retrieved


class ViewFusion(nn.Module):
    """관점별 노드 임베딩을 하나로 합친다.

    0번(local)을 기준으로 두고 나머지 관점은 **local 임베딩에서 선형으로 만든 게이트**를
    거쳐 더한다::

        g[b,n,:] = Linear(local[b,n])            # [V-1] — 노드·샘플마다 다르다
        fused    = local + sum_j g[b,n,j] * view_j[b,n]

    "지금 이 노드의 국소 상황이 이렇다면 다른 관점을 얼마나 믿을까"를 local이 정한다.
    Linear의 weight와 bias를 0으로 초기화하므로 학습 시작 시점의 함수는 local 단독과
    정확히 같고, 게이트는 쓸모 있는 만큼만 자란다.
    """

    def __init__(self, config: MergedDemandConfig, num_views: int) -> None:
        super().__init__()
        if num_views < 1:
            raise ValueError('num_views는 1 이상이어야 함')
        self.num_views = num_views
        # 0번 관점은 게이트 없이 그대로 간다. 나머지 V-1개의 게이트를 local에서 만든다.
        self.gate_proj: nn.Linear | None = None
        if num_views > 1:
            self.gate_proj = nn.Linear(config.d_model, num_views - 1)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

    def gate(self, local: Tensor) -> Tensor | None:
        """``[B, N, D]`` -> ``[B, N, V-1]`` 관점별 게이트. 관점이 하나뿐이면 None."""

        if self.gate_proj is None:
            return None
        return self.gate_proj(local)

    def forward(self, views: Tensor) -> Tensor:
        """
        Args:
            views: ``[B, N, V, D]`` 관점별 노드 임베딩. 0번은 local이다.

        Returns:
            ``[B, N, D]`` 융합된 노드 임베딩.
        """

        if views.shape[2] != self.num_views:
            raise ValueError(f'관점 수가 {self.num_views}가 아님: {views.shape[2]}')
        local = views[:, :, 0]
        gate = self.gate(local)
        if gate is None:
            return local
        return local + (gate.unsqueeze(-1) * views[:, :, 1:]).sum(dim=2)


class PredictionHead(nn.Module):
    """융합된 노드 임베딩 -> 그 노드의 다음 1시간 수요(스칼라).

    수요는 음수가 될 수 없으므로 마지막에 비음수 활성(Softplus 등)을 씌운다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, fused: Tensor) -> Tensor:
        """
        Args:
            fused: ``[B, N, D]``

        Returns:
            ``[B, N]`` 노드별 예측 수요(raw 스케일, >= 0).
        """

        # softplus(x) = log(1+exp(x)). 수요는 음수가 될 수 없고, ReLU와 달리 0 근처에서
        # gradient가 죽지 않는다 — 셀의 74%가 0인 데이터라 이 차이가 중요하다.
        return F.softplus(self.mlp(fused).squeeze(-1))


class MergedDemandModel(PreTrainedModel):
    """네 관점(local / daily / weekly / retrieval)을 융합하는 수요 예측 모델."""

    config_class = MergedDemandConfig
    base_model_prefix = 'merged_demand'
    main_input_name = 'demand_history'

    # ViewFusion에 들어가는 임베딩 관점의 순서. 검색은 여기 없다 — raw 예측값이라
    # 임베딩 공간이 아니라 NeuralRetrievalGate가 출력 단계에서 섞는다.
    # daily/weekly는 아직 인코더가 비어 있어 살아 있는 임베딩 관점은 local 하나다.
    VIEW_NAMES = ('local',)

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__(config)

        self.height = config.height
        self.width = config.width

        # 학습에서 제외할 노드(train 구간 평균 수요가 거의 0). 예측을 정확히 0으로 고정하고
        # 손실에서도 빼서, 남은 노드에만 용량과 gradient가 가도록 한다.
        # 텐서가 아니라 파이썬 리스트로 들고 있는다 — from_pretrained가 meta device에서
        # __init__을 돌기 때문에 버퍼로 만들면 체크포인트에 없을 때 쓰레기값이 남는다.
        self.zero_node_indices = (
            [int(v) for v in config.zero_node_indices] if config.zero_node_indices else None
        )
        if self.zero_node_indices is not None:
            bad = [i for i in self.zero_node_indices if not 0 <= i < config.num_nodes]
            if bad:
                raise ValueError(f'zero_node_indices가 노드 범위(0..{config.num_nodes - 1})를 벗어남: {bad[:5]}')
            if len(self.zero_node_indices) >= config.num_nodes:
                raise ValueError('모든 노드를 학습에서 제외할 수는 없다')
        self._cached_node_keep: Tensor | None = None

        self.context_encoder = ContextEncoder(config)
        self.local_view = LocalViewEncoder(config)
        self.local_radius = config.local_radius
        # 검색은 원본 grid 경로가 있고 use_retrieval이 켜져 있을 때만 만든다. 둘 중 하나라도
        # 빠지면 검색이 없던 때와 완전히 같은 모델이다.
        self.use_retrieval = bool(config.use_retrieval and config.retrieval_grid_path)
        self.retrieval: CausalRetrieval | None = None
        self.output_gate: NeuralRetrievalGate | None = None
        if self.use_retrieval:
            self.retrieval = CausalRetrieval(config)
            self.output_gate = NeuralRetrievalGate(config)
        self.view_names = self.VIEW_NAMES
        self.fusion = ViewFusion(config, num_views=len(self.view_names))
        self.head = PredictionHead(config)

        self.loss_fn: nn.Module
        self.configure_loss(config.loss_type)
        self.post_init()

    def configure_loss(self, loss_type: str) -> None:
        """학습 loss를 교체하고 선택값을 config에도 동기화한다.

        config를 함께 갱신해야 체크포인트를 ``from_pretrained``로 다시 열었을 때 런타임에
        골랐던 loss가 조용히 되돌아가지 않는다.
        """

        self.loss_fn = build_loss(
            loss_type,
            gamma=self.config.loss_gamma,
            eps=self.config.loss_eps,
            rmse_weight=self.config.rmse_weight,
            split_threshold=self.config.split_threshold,
            split_high_weight=self.config.split_high_weight,
        )
        # rmse_mape처럼 요소별로 분해되지 않는 손실은 집계 방식이 다르다.
        self.loss_is_scalar = loss_type in SCALAR_LOSS_TYPES
        # ``PreTrainedModel.__init__``도 같은 이름의 속성을 쓰므로 property로 만들면 안 된다.
        self.loss_type = loss_type
        self.config.loss_type = loss_type

    def _init_weights(self, module: nn.Module) -> None:
        """PyTorch 기본 초기화를 그대로 쓴다 — 의도적으로 아무것도 하지 않는다.

        ``PreTrainedModel._init_weights``는 Linear/Embedding/LayerNorm을 std=0.02
        정규분포로 다시 초기화한다. 여기 블록들은 전부 PyTorch 기본 초기화(fan-in 기반
        uniform)를 전제로 설계됐고, 특히 ``node_embedding``/``position_embedding``은
        생성자에서 직접 std=0.02를 주고 있다. ``super()._init_weights(module)``를 부르면
        그 의도가 조용히 덮인다.
        """

        return

    def node_keep_mask(self, device: torch.device) -> Tensor | None:
        """``[H, W]``. 학습에 쓰는 노드가 1.0, 제외한 노드가 0.0. 제외가 없으면 None.

        config의 파이썬 리스트에서 만들되 device별로 캐시한다.
        """

        if self.zero_node_indices is None:
            return None
        cached = self._cached_node_keep
        if cached is None or cached.device != device:
            keep = torch.ones(self.height * self.width, device=device)
            keep[torch.as_tensor(self.zero_node_indices, dtype=torch.long, device=device)] = 0.0
            cached = keep.view(self.height, self.width)
            self._cached_node_keep = cached
        return cached

    def forward(
        self,
        demand_history: Tensor,
        daily_demand: Tensor,
        daily_mask: Tensor,
        weekly_demand: Tensor,
        weekly_mask: Tensor,
        sample_idx: Tensor,
        weather: Tensor,
        hour_of_day: Tensor,
        day_of_week: Tensor,
        daily_weather: Tensor,
        daily_hour: Tensor,
        daily_day_of_week: Tensor,
        weekly_weather: Tensor,
        weekly_hour: Tensor,
        weekly_day_of_week: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """인자 이름은 ``UnifiedDemandDataset.__getitem__``의 키와 1:1로 대응한다.

        최근 k시간 (예측 대상 시점 t 기준 ``[t-k, t)``):
            demand_history: ``[B, k, H, W]`` 격자 전체 수요(raw).
            weather: ``[B, k, 3]``  — 한 칸 밀린 ``[t-k+1, t+1)``이다. 예측 시점의 날씨까지
                포함하는 의도적 설계("단기 날씨 예보는 이미 안다").
            hour_of_day / day_of_week: ``[B, k]`` 정수.

        일 주기 lag (24, 48, ... 시간 전):
            daily_demand: ``[B, Ld, N, 1]`` / daily_mask: ``[B, Ld]`` bool, **True가 무효**.
            daily_weather: ``[B, Ld, 3]`` / daily_hour, daily_day_of_week: ``[B, Ld]``.

        주 주기 lag (168, 336, ... 시간 전):
            weekly_demand: ``[B, Lw, N, 1]`` / weekly_mask: ``[B, Lw]``.
            weekly_weather: ``[B, Lw, 3]`` / weekly_hour, weekly_day_of_week: ``[B, Lw]``.

        그 외:
            sample_idx: ``[B]`` 예측 대상 시점 t의 절대 인덱스. 검색의 인과성 경계.
            labels: ``[B, H, W]`` 시점 t의 실제 수요. 없으면 손실을 계산하지 않는다.

        Returns:
            ``{'logits': [B, H, W]}``. ``labels``가 있으면 ``'loss'``(스칼라)가 추가된다.
            ``Trainer``는 ``loss``를 제외한 모든 키를 배치마다 gather하므로 디버그 텐서를
            여기 넣으면 안 된다 — 필요하면 :meth:`forward_views`를 쓴다.

        **daily/weekly 관점은 아직 비어 있다.** 그 인자들은 받기만 하고 읽지 않는다 —
        인코더를 채울 때 dataset/train.py를 건드리지 않으려고 목록을 미리 맞춰 둔 것이다.
        local과 retrieval 두 관점이 ``ViewFusion``으로 합쳐진다.
        """

        views = self._encode_views(
            demand_history=demand_history,
            weather=weather,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
        )
        # [B, N_active] — 학습에서 뺀 노드가 있으면 그만큼 짧다.
        predictions = self.head(views['fused'])
        active = self.local_view.active_node_index(predictions.device)
        if self.retrieval is not None and self.output_gate is not None:
            local_crop = crop_local_windows(demand_history, self.local_radius)
            retrieved = self.retrieval(local_crop, sample_idx)  # [B, N] raw 스케일
            if active is not None:
                retrieved = retrieved.index_select(1, active)
            _, predictions = self.output_gate(views['fused'], predictions, retrieved)
        if active is None:
            logits = predictions.reshape(-1, self.height, self.width)
        else:
            # 뺀 노드 자리는 **정확히** 0이다. Softplus는 0에 점근할 뿐 0이 될 수 없으므로
            # 계산해서 버리는 대신 아예 자리만 0으로 채운다. autograd가 두 경로의 gradient를
            # 받아야 하므로 in-place 대입이 아니라 out-of-place index_copy를 쓴다.
            full = predictions.new_zeros(len(predictions), self.height * self.width)
            logits = full.index_copy(1, active, predictions).reshape(
                -1, self.height, self.width
            )
        keep = self.node_keep_mask(logits.device)

        output: dict[str, Tensor] = {'logits': logits}
        if labels is not None:
            if self.loss_is_scalar:
                # rmse_mape는 전체 원소에 걸친 하나의 sqrt(mean(...))이라 요소별로 분해되지
                # 않는다. 손실 객체가 이미 스칼라를 돌려준다.
                if keep is not None:
                    raise ValueError(
                        f'loss_type={self.loss_type!r}는 요소별로 분해되지 않아 노드 제외 '
                        '마스크를 적용할 수 없다 — combined/mae/demand_split 중에서 고른다'
                    )
                output['loss'] = self.loss_fn(logits, labels)
            else:
                # reduction='none'으로 받아 여기서 한 번만 평균낸다. 배치 크기가 균일하지
                # 않아(drop_last=False) 배치 평균의 평균은 원소 평균과 다르다.
                elementwise = self.loss_fn(logits, labels)
                if keep is None:
                    output['loss'] = elementwise.mean()
                else:
                    # 제외 노드의 오차는 gradient를 만들지 않는다. 남은 노드에 대해서만
                    # 평균내야 제외 노드 수가 손실의 크기(=유효 학습률)를 바꾸지 않는다.
                    output['loss'] = (elementwise * keep).sum() / (keep.sum() * len(elementwise))
        return output

    def _encode_views(
        self,
        *,
        demand_history: Tensor,
        weather: Tensor,
        hour_of_day: Tensor,
        day_of_week: Tensor,
    ) -> dict[str, Tensor]:
        """임베딩 관점과 융합 결과. ``forward``와 ``forward_views``가 공유한다.

        지금은 관점이 local 하나뿐이라 ``fused``가 ``local``과 같다. 검색은 여기 들어오지
        않는다 — 임베딩이 아니라 raw 예측값이라 출력 단계에서 섞인다.
        """

        context = self.context_encoder(weather, hour_of_day, day_of_week)  # [B,k,C]
        encoded = {'local': self.local_view(demand_history, context)}  # [B,N,D]
        stacked = torch.stack([encoded[name] for name in self.view_names], dim=2)
        encoded['fused'] = self.fusion(stacked)
        return encoded

    def forward_views(self, **batch) -> dict[str, Tensor]:
        """관점별 임베딩과 중간 값을 돌려준다(분석/디버깅용, 학습 경로에서 쓰지 않는다).

        Returns:
            ``{'local', 'fused', 'neural_pred', 'logits'}``. 검색이 켜져 있으면
            ``'retrieved'``(검색 예측)와 ``'lambda_weight'``(뉴럴 쪽 신뢰도)가 함께 나온다.
        """

        views = self._encode_views(
            demand_history=batch['demand_history'],
            weather=batch['weather'],
            hour_of_day=batch['hour_of_day'],
            day_of_week=batch['day_of_week'],
        )
        neural_pred = self.head(views['fused'])
        views['neural_pred'] = neural_pred
        predictions = neural_pred
        if self.retrieval is not None and self.output_gate is not None:
            local_crop = crop_local_windows(batch['demand_history'], self.local_radius)
            retrieved = self.retrieval(local_crop, batch['sample_idx'])
            active = self.local_view.active_node_index(neural_pred.device)
            if active is not None:
                retrieved = retrieved.index_select(1, active)
            lambda_weight, predictions = self.output_gate(
                views['fused'], neural_pred, retrieved
            )
            views['retrieved'] = retrieved
            views['lambda_weight'] = lambda_weight
        views['logits'] = predictions.reshape(-1, self.height, self.width)
        return views


__all__ = [
    'CausalRetrieval',
    'ContextEncoder',
    'FourierScalarEmbedding',
    'LocalViewEncoder',
    'MergedDemandModel',
    'NeuralRetrievalGate',
    'PeriodicViewEncoder',
    'PredictionHead',
    'ViewFusion',
    'crop_local_windows',
    'make_neighbor_valid',
]
