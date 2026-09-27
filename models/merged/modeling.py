"""Local-view demand forecasting with optional daily/weekly periodic branches.

``periodic_mode='none'`` preserves the local-only model. ``lstm`` encodes valid
daily/weekly lags as D-vectors, concatenates them with the local D-vector and
uses the existing prediction head. ``ma`` averages the raw center-node history
alongside daily/weekly demand with a learned convex node-wise mixture; ``ema``
averages periodic demand with independent node gates. ``lag_lstm`` replaces the
periodic mean with an input-1 hidden-4 LSTM and scalar projection.
The retrieval branch is not implemented or used in these experiments.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import PreTrainedModel

from .config import MergedDemandConfig
from .losses import SCALAR_LOSS_TYPES, build_loss


class ContextEncoder(nn.Module):
    """시점별 보조 정보(날씨 + 캘린더)를 한 벡터로 묶는다.

    local 및 LSTM 주기 관점이 같은 날씨/캘린더 임베딩을 쓴다.
    MA/EMA 주기 관점은 원 수요만 평균내므로 이 context를 쓰지 않는다.

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
        self.use_weather = getattr(config, 'use_weather', True)
        self.use_calendar = getattr(config, 'use_calendar', True)

    def forward(self, weather: Tensor, hour_of_day: Tensor, day_of_week: Tensor) -> Tensor:
        """
        Args:
            weather: ``[B, L, 3]`` 원본 스케일 (기온, 강수량, 적설).
            hour_of_day: ``[B, L]`` 0..23 정수.
            day_of_week: ``[B, L]`` 0..6 정수.

        Returns:
            ``[B, L, C]`` 시점별 보조 정보.
        """

        if self.use_weather:
            temperature, rainfall, snowfall = weather.unbind(dim=-1)
            # Train-only extrema; disabling weather removes the entire two-channel signal.
            temperature_norm = (temperature - self.temperature_min) / self.temperature_range
            precipitation = rainfall + self.snow_scale * snowfall
            precipitation_norm = precipitation / self.precipitation_max
            weather_features = torch.stack([temperature_norm, precipitation_norm], dim=-1)
        else:
            weather_features = weather.new_zeros(*weather.shape[:-1], 2)

        if self.use_calendar:
            weekday_features = self.weekday_embedding(day_of_week)
            hour_features = self.hour_embedding(hour_of_day)
        else:
            weekday_features = weather.new_zeros(*weather.shape[:-1], self.weekday_embedding.embedding_dim)
            hour_features = weather.new_zeros(*weather.shape[:-1], self.hour_embedding.embedding_dim)
        return torch.cat([weather_features, weekday_features, hour_features], dim=-1)


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


class WindowConvEncoder(nn.Module):
    """창 하나를 attention 대신 2D conv로 요약한다.

    ``(2a+1) x (2a+1)`` 창을 채널이 ``D``인 이미지로 보고 residual conv 블록을 쌓는다.
    블록 구조는 Transformer의 pre-norm 잔차 블록과 같게 맞췄다::

        x = x + Dropout(GELU(Conv3x3(LayerNorm(x))))

    위치 임베딩이 없다 — conv 커널 자체가 "어느 방향 이웃인지"를 가중치로 구분하므로
    Transformer처럼 자리를 따로 알려 줄 필요가 없다. 요약은 CLS 토큰이 아니라 창의
    **중앙 칸**(=그 노드 자신)에서 읽는다.
    """

    def __init__(self, d_model: int, window_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.center = (window_size * window_size) // 2
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self.convs = nn.ModuleList(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1) for _ in range(layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """``[M, P, D]`` 창 토큰 -> ``[M, D]`` 창 요약."""

        size = self.window_size
        batch, neighbors, d_model = tokens.shape
        if neighbors != size * size:
            raise ValueError(f'창 토큰 수가 {size * size}가 아님: {neighbors}')
        hidden = tokens
        for norm, conv in zip(self.norms, self.convs):
            normed = norm(hidden)
            grid = normed.transpose(1, 2).reshape(batch, d_model, size, size)
            delta = conv(grid).reshape(batch, d_model, neighbors).transpose(1, 2)
            hidden = hidden + self.dropout(F.gelu(delta))
        return hidden[:, self.center]


class LocalViewEncoder(nn.Module):
    """관점 1 — 공간. "내 주변 (2a+1)^2 칸에서 최근 k시간 동안 무슨 일이 있었나."

    유일하게 이웃 노드의 수요를 보는 관점이다. 나머지 세 관점은 노드별로 독립이다.

    시간축에 평활화를 켜면 노드별 공유 valid c탭 가중합을 crop 전에 적용해
    ``k'=k-c+1``로 줄인다. 끄면 ``k'=k``. 이후 두 단계로 접는다::

        [B,k,H,W] --valid smoothing--> [B,k',H,W]
                  --crop--> [B,k',N,P]       노드별 창
                  --embed--> [B,k',N,1+P,D] 창 안의 각 칸이 토큰 하나, 맨 앞은 CLS
                  --transformer + CLS --> [B,k',N,D] (공간 축을 접음)
                  --LSTM --> [B,N,D]        (시간 축을 접음)

    Transformer는 **창 하나**를 본다 — (배치, 시점, 노드)를 전부 batch 축으로 접으므로
    ``B*k'*N``개의 길이 ``1+P`` 시퀀스가 한 번에 들어간다. 창 안의 공간 배치는
    attention이 직접 알 수 없고 ``position_embedding``이 학습으로 담는다.

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
        self.use_neighbors = getattr(config, 'use_neighbors', True)

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
        # 양수·합 1 제약으로 학습 중에도 평균 역할을 유지한다. 설정이 없으면 기존
        # 모델의 state_dict와 연산을 그대로 보존한다.
        self.history_logits = (
            nn.Parameter(torch.tensor([math.log(value) for value in config.history_weights]))
            if config.history_weights is not None else None
        )
        # 0 = CLS(요약을 모으는 자리), 1 = EDGE(격자 밖 = 값이 없음).
        self.special_embedding = nn.Embedding(2, d_model)
        # "몇 번 노드의 요약인지"를 창에 새겨 넣는다. transformer 경로는 CLS 토큰에,
        # conv 경로는 창의 모든 칸에 더한다(CLS가 없으므로).
        self.node_embedding = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.local_encoder = str(config.local_encoder)
        if self.local_encoder not in ('transformer', 'conv'):
            raise ValueError("local_encoder는 'transformer' | 'conv'여야 함")
        if self.local_encoder == 'transformer':
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
        else:
            # conv 경로에는 CLS도 위치 임베딩도 없다. 노드 정체성은 창의 모든 칸에
            # 브로드캐스트로 더하고, 요약은 중앙 칸에서 읽는다.
            self.window_conv = WindowConvEncoder(
                d_model,
                window_size=config.window_size,
                layers=config.transformer_layers,
                dropout=config.dropout,
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
        # 기존 파라미터의 시드별 초기화를 보존하도록 추가 층의 초기화 RNG를 격리한다.
        self.inter_node_transformer: nn.TransformerEncoder | None = None
        if config.use_inter_node_transformer:
            with torch.random.fork_rng(devices=[]):
                inter_node_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=config.transformer_heads,
                    dim_feedforward=config.transformer_ffn,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                inter_node_layer.self_attn.dropout = config.attention_dropout
                self.inter_node_transformer = nn.TransformerEncoder(
                    inter_node_layer, num_layers=1, enable_nested_tensor=False,
                )

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
        if self.history_logits is not None:
            batch, steps, height, width = demand_history.shape
            kernel_size = self.history_logits.numel()
            if steps < kernel_size:
                raise ValueError(f'history 길이 {steps}가 kernel 길이 {kernel_size}보다 짧음')
            # conv1d는 cross-correlation: 오래된 값부터 최신 값까지 [w1,...,wc].
            # 격자마다 동일한 kernel을 쓰고 패딩 없이 k-c+1 시점으로 줄인다.
            series = demand_history.permute(0, 2, 3, 1).reshape(-1, 1, steps)
            kernel = F.softmax(self.history_logits, dim=0).view(1, 1, kernel_size)
            demand_history = F.conv1d(series, kernel).reshape(
                batch, height, width, steps - kernel_size + 1
            ).permute(0, 3, 1, 2)
            # 각 합성 시점의 마지막 수요 시점에 해당하는 원래 context를 사용한다.
            context = context[:, kernel_size - 1:]
        local_crop = crop_local_windows(demand_history, self.local_radius)
        # 여기서 자른다 — crop이 끝난 뒤라 뺀 노드도 남은 노드의 창 안에는 그대로 들어 있다.
        active = self.active_node_index(local_crop.device)
        neighbor_valid = self.neighbor_valid
        if active is not None:
            local_crop = local_crop.index_select(2, active)
            neighbor_valid = neighbor_valid.index_select(0, active)
        if not self.use_neighbors:
            # Zero only other nodes' demand; the center and the spatial layout stay intact.
            center = local_crop.shape[-1] // 2
            local_crop = local_crop.clone()
            local_crop[..., :center] = 0
            local_crop[..., center + 1:] = 0
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

        if self.local_encoder == 'transformer':
            cls_token = self.special_embedding.weight[0].view(1, 1, 1, 1, -1)
            cls_token = cls_token + node_embedding.view(1, 1, nodes, 1, -1)
            cls_token = cls_token.expand(batch, steps, -1, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=3)
            tokens = tokens + self.position_embedding.view(1, 1, 1, 1 + neighbors, -1)
            # (배치, 시점, 노드)를 batch 축으로 접는다 — 창 하나가 시퀀스 하나다.
            encoded = self.transformer(
                tokens.reshape(batch * steps * nodes, 1 + neighbors, -1)
            )
            summary = encoded[:, 0].reshape(batch, steps, nodes, -1)  # CLS만 꺼낸다
        else:
            # CLS 자리가 없으므로 노드 임베딩을 창 전체에 더한다. 요약은 중앙 칸이다.
            tokens = tokens + node_embedding.view(1, 1, nodes, 1, -1)
            encoded = self.window_conv(
                tokens.reshape(batch * steps * nodes, neighbors, -1)
            )
            summary = encoded.reshape(batch, steps, nodes, -1)
        if self.inter_node_transformer is not None:
            # 같은 시각의 N개 패치 CLS를 토큰으로 사용한다. 시간축 LSTM과 이후 융합은 동일하다.
            summary = self.inter_node_transformer(
                summary.reshape(batch * steps, nodes, -1)
            ).reshape(batch, steps, nodes, -1)

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
    """일/주 lag를 LSTM D벡터, MA/EMA 평균 또는 LSTM 스칼라로 접는다."""

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        self.mode = config.periodic_mode
        if config.periodic_window_weights is not None:
            # 두 encoder(daily/weekly)가 각자 학습한다. RNG를 보존해 다른 모듈의
            # 같은 seed 초기화는 비평활화 기준선과 같게 유지한다.
            with torch.random.fork_rng(devices=[]):
                self.window_projection = nn.Linear(5, 1, bias=False)
            with torch.no_grad():
                self.window_projection.weight.copy_(
                    torch.tensor(config.periodic_window_weights).unsqueeze(0)
                )
        else:
            self.window_projection = None
        if self.mode == 'lstm':
            self.lstm = nn.LSTM(1 + config.context_dim, config.d_model, batch_first=True)
        elif self.mode == 'lag_lstm':
            # 각 관점별 5시간 창 결합 이후, lag 간 MA를 별도 소형 LSTM으로 대체한다.
            self.lag_lstm = nn.LSTM(1, 4, batch_first=True)
            self.lag_projection = nn.Linear(4, 1)

    def forward(self, lag_demand: Tensor, lag_mask: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """[B,L,N,1], 무효=True인 [B,L] -> ([B,N,D] 또는 [B,N], 유효=[B])."""
        if lag_demand.ndim != 4 or lag_demand.shape[-1] != 1:
            raise ValueError(f'lag_demand는 [B,L,N,1]이어야 함: {tuple(lag_demand.shape)}')
        batch, length, nodes, _ = lag_demand.shape
        if lag_mask.shape != (batch, length):
            raise ValueError(f'lag_mask는 [B,L]이어야 함: {tuple(lag_mask.shape)}')
        if self.window_projection is not None:
            window_size = self.window_projection.in_features
            if length % window_size:
                raise ValueError(f'lag 길이 {length}가 window 크기 {window_size}의 배수가 아님')
            groups = length // window_size
            # _chronological_lags의 순서: 먼 쪽 2시간, 1시간, 중심, 가까운 쪽 1·2시간.
            # 시작 전 데이터가 끼어 있는 묶음은 0으로 패딩해 기여시키지 않는다.
            group_valid = (~lag_mask.bool()).reshape(batch, groups, window_size).all(-1)
            windows = lag_demand.reshape(batch, groups, window_size, nodes).permute(0, 1, 3, 2)
            lag_demand = self.window_projection(windows)
            lag_mask = ~group_valid
            lag_demand = lag_demand.masked_fill(lag_mask[:, :, None, None], 0)
            if context is not None:
                context = context[:, window_size // 2::window_size]
            length = groups
        valid = ~lag_mask.bool()
        lengths = valid.sum(dim=1)
        if self.mode in ('ma', 'ma_no_local', 'ema'):
            weights = valid.to(lag_demand.dtype)
            if self.mode == 'ema':
                # 가장 가까운 lag가 마지막이다. 유효 lag만 다시 정규화해 상수 수요를 보존한다.
                alpha = 2.0 / (length + 1)
                age = torch.arange(length - 1, -1, -1, device=lag_demand.device)
                weights = weights * (1.0 - alpha) ** age
            average = torch.einsum('bln,bl->bn', lag_demand.squeeze(-1), weights)
            average = average / weights.sum(dim=1).clamp_min(1e-12).unsqueeze(-1)
            return average, lengths > 0

        if self.mode == 'lag_lstm':
            # MA와 동일한 raw 스칼라 입력; 날씨/캘린더, log1p는 추가하지 않는다.
            sequence = lag_demand
            recurrent = self.lag_lstm
        elif self.mode == 'lstm':
            if context is None or context.shape != (batch, length, self.lstm.input_size - 1):
                raise ValueError(f'context는 [B,L,{self.lstm.input_size - 1}]이어야 함')
            sequence = torch.log1p(torch.clamp_min(lag_demand, 0))
            sequence = torch.cat(
                [sequence, context[:, :, None, :].expand(-1, -1, nodes, -1)], dim=-1
            )
            recurrent = self.lstm
        else:
            raise ValueError(f'구현되지 않은 periodic_mode: {self.mode}')
        # 원본 lag 순서를 보존하며 무효 시점을 뒤로 보낸다. 패킹에 앞서 노드마다 펼친다.
        order = valid.long().argsort(dim=1, descending=True, stable=True)
        sequence = sequence.gather(
            1, order[:, :, None, None].expand(-1, -1, nodes, sequence.shape[-1])
        )
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * nodes, length, -1)
        row_lengths = lengths.repeat_interleave(nodes)
        active = (row_lengths > 0).nonzero(as_tuple=True)[0]
        output = sequence.new_zeros(batch * nodes, 1 if self.mode == 'lag_lstm' else recurrent.hidden_size)
        if active.numel():
            packed = pack_padded_sequence(
                sequence.index_select(0, active),
                row_lengths.index_select(0, active).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = recurrent(packed)
            encoded = self.lag_projection(hidden[-1]) if self.mode == 'lag_lstm' else hidden[-1]
            output = output.index_copy(0, active, encoded)
        if self.mode == 'lag_lstm':
            return output.reshape(batch, nodes), lengths > 0
        return output.reshape(batch, nodes, -1), lengths > 0


class RetrievalViewEncoder(nn.Module):
    """관점 4 — 유추. "지금과 비슷했던 과거 n개 시점에서는, 그 다음 시간에 무슨 일이 있었나."

    두 단계로 나뉜다:

    1. :meth:`search` — 학습 대상이 아니다(``no_grad``). 원본 temporal grid를 직접 읽어
       현재 윈도우와 가장 비슷한 과거 시점 n개를 고른다. 인과성을 지켜야 하므로 후보는
       ``tau < target_time``(``retrieval_scope='observed_past'``) 또는 train split 안쪽
       (``'train_prefix'``)으로 제한한다.
    2. :meth:`forward` — 그렇게 찾아온 값들을 다른 세 관점과 같은 ``[B, N, D]`` 임베딩으로
       바꾼다. **이전 설계와 달라진 지점이 여기다** — 검색 결과를 곧바로 예측값(스칼라)으로
       쓰지 않고, 관점 하나의 임베딩으로 만들어 융합 단계에 넘긴다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()

    def search(self, demand_history: Tensor, sample_idx: Tensor) -> tuple[Tensor, Tensor]:
        """현재 윈도우와 비슷한 과거 시점 n개를 찾아 그 **직후 실제 수요**를 가져온다.

        학습 파라미터가 없고 gradient도 흐르지 않는다. 같은 ``sample_idx``에 대한 결과는
        항상 같으므로 캐시할 수 있다.

        Args:
            demand_history: ``[B, k, H, W]`` 질의로 쓸 최근 k시간 윈도우.
            sample_idx: ``[B]`` 예측 대상 시점 t의 **절대** 시간 인덱스(split 로컬 아님).
                인과성 경계를 정하는 데 쓴다.

        Returns:
            ``neighbor_demand``: ``[B, n, N]`` 찾아온 각 시점 tau의 **다음 시간** 실제 수요.
                즉 "그때는 이렇게 됐다"는 답안지다.
            ``neighbor_score``: ``[B, n]`` 질의와의 유사도. 클수록 비슷하다.
        """

        pass

    def forward(self, demand_history: Tensor, sample_idx: Tensor) -> Tensor:
        """
        Args:
            demand_history: ``[B, k, H, W]``
            sample_idx: ``[B]``

        Returns:
            ``[B, N, D]`` 노드별 유추 관점 임베딩.
        """

        pass


class ViewFusion(nn.Module):
    """LSTM: concatenate three D-vectors; scalar modes: directly predict demand."""

    def __init__(self, config: MergedDemandConfig, num_views: int) -> None:
        super().__init__()
        self.mode = config.periodic_mode
        self.use_local_view = getattr(config, 'use_local_view', True)
        self.use_local_mean = getattr(config, 'use_local_mean', True)
        self.use_softplus = getattr(config, 'use_softplus', True)
        if self.mode == 'lstm':
            self.projection = nn.Linear(num_views * config.d_model, config.d_model)
        elif self.mode in ('ma', 'ma_no_local', 'ema', 'lag_lstm'):
            self.local_projection = nn.Linear(config.d_model, 1)
            if self.mode == 'ma':
                # Local is the reference logit (0): softmax starts at 0.7/0.2/0.1.
                self.daily_mix_logit = nn.Parameter(
                    torch.full((config.num_nodes,), math.log(0.2 / 0.7))
                )
                self.weekly_mix_logit = nn.Parameter(
                    torch.full((config.num_nodes,), math.log(0.1 / 0.7))
                )
            else:
                # Unchanged scalar variants: two independent sigmoid gates.
                self.daily_gate = nn.Parameter(torch.zeros(config.num_nodes))
                self.weekly_gate = nn.Parameter(torch.zeros(config.num_nodes))

    def forward(
        self, local: Tensor, daily: Tensor, weekly: Tensor,
        daily_valid: Tensor, weekly_valid: Tensor, node_index: Tensor | None = None,
        local_mean: Tensor | None = None,
    ) -> Tensor:
        if self.mode == 'lstm':
            daily = daily * daily_valid[:, None, None]
            weekly = weekly * weekly_valid[:, None, None]
            return self.projection(torch.cat([local, daily, weekly], dim=-1))
        daily = daily * daily_valid[:, None]
        weekly = weekly * weekly_valid[:, None]
        local_term = (
            self.local_projection(local).squeeze(-1)
            if self.use_local_view else local.new_zeros(local.shape[:2])
        )
        if self.mode == 'ma':
            if self.use_local_mean and local_mean is None:
                raise ValueError('MA 융합에는 중앙 노드 local 수요 평균이 필요함')
            daily_logit = (
                self.daily_mix_logit if node_index is None
                else self.daily_mix_logit.index_select(0, node_index)
            )
            weekly_logit = (
                self.weekly_mix_logit if node_index is None
                else self.weekly_mix_logit.index_select(0, node_index)
            )
            local_gate, daily_gate, weekly_gate = torch.stack(
                (torch.zeros_like(daily_logit), daily_logit, weekly_logit)
            ).softmax(dim=0).unbind(0)
            score = local_term
            if self.use_local_mean:
                score = score + local_gate * local_mean
            score = score + daily_gate * daily + weekly_gate * weekly
        else:
            daily_gate = self.daily_gate if node_index is None else self.daily_gate.index_select(0, node_index)
            weekly_gate = self.weekly_gate if node_index is None else self.weekly_gate.index_select(0, node_index)
            score = (
                local_term
                + daily_gate.sigmoid().unsqueeze(0) * daily
                + weekly_gate.sigmoid().unsqueeze(0) * weekly
            )
        return F.softplus(score) if self.use_softplus else score


class PredictionHead(nn.Module):
    """융합된 노드 임베딩 -> 그 노드의 다음 1시간 수요(스칼라).

    기본 설정은 비음수 Softplus 출력이며, use_softplus=false는 활성 없이 내보낸다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.use_softplus = getattr(config, 'use_softplus', True)

    def forward(self, fused: Tensor) -> Tensor:
        """
        Args:
            fused: ``[B, N, D]``

        Returns:
            ``[B, N]`` 노드별 예측 수요(raw 스케일). 기본은 >=0.
        """

        # 기본 Softplus는 0 근처에서도 gradient가 살아 있다(많은 수요 셀이 0).
        # 출력 활성 ablation은 음수 예측도 허용한다.
        prediction = self.mlp(fused).squeeze(-1)
        return F.softplus(prediction) if self.use_softplus else prediction


class MergedDemandModel(PreTrainedModel):
    """Local demand predictor with optional daily/weekly periodic branches."""

    config_class = MergedDemandConfig
    base_model_prefix = 'merged_demand'
    main_input_name = 'demand_history'

    VIEW_NAMES = ('local', 'daily', 'weekly')

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
        self.periodic_mode = config.periodic_mode
        # 기존 local-only의 파라미터·초기화 순서를 보존한다.
        if self.periodic_mode != 'none':
            self.daily_view = PeriodicViewEncoder(config)
            self.weekly_view = PeriodicViewEncoder(config)
        self.retrieval_view = RetrievalViewEncoder(config)
        self.fusion = ViewFusion(config, num_views=len(self.VIEW_NAMES))
        if self.periodic_mode not in ('ma', 'ma_no_local', 'ema', 'lag_lstm'):
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

        periodic_mode='none'이면 기존 local-only 경로다. 'lstm'은 daily/weekly D벡터를
        local과 융합한다. 'ma'는 local 원수요 평균까지 3-way 혼합하고
        'ma_no_local'은 과거의 독립 daily/weekly sigmoid 경로를 재현한다.
        'ema'/'lag_lstm'도 독립 sigmoid 게이트로 주기 출력을 예측에 더한다.
        retrieval 입력은 아직 사용하지 않는다.
        use_softplus=false에서는 logits가 음수일 수 있다.
        """

        views = self._encode_views(
            demand_history=demand_history,
            daily_demand=daily_demand,
            daily_mask=daily_mask,
            weekly_demand=weekly_demand,
            weekly_mask=weekly_mask,
            weather=weather,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            daily_weather=daily_weather,
            daily_hour=daily_hour,
            daily_day_of_week=daily_day_of_week,
            weekly_weather=weekly_weather,
            weekly_hour=weekly_hour,
            weekly_day_of_week=weekly_day_of_week,
        )
        logits = self._predict_grid(views)
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

    def _predict_grid(self, views: dict[str, Tensor]) -> Tensor:
        predictions = (
            views['prediction'] if self.periodic_mode in ('ma', 'ma_no_local', 'ema', 'lag_lstm')
            else self.head(views['fused'])
        )
        active = self.local_view.active_node_index(predictions.device)
        if active is None:
            return predictions.reshape(-1, self.height, self.width)
        # 제외 노드는 Softplus 결과 대신 정확히 0으로 남긴다.
        full = predictions.new_zeros(len(predictions), self.height * self.width)
        return full.index_copy(1, active, predictions).reshape(-1, self.height, self.width)

    def _encode_views(
        self,
        *,
        demand_history: Tensor,
        daily_demand: Tensor,
        daily_mask: Tensor,
        weekly_demand: Tensor,
        weekly_mask: Tensor,
        weather: Tensor,
        hour_of_day: Tensor,
        day_of_week: Tensor,
        daily_weather: Tensor,
        daily_hour: Tensor,
        daily_day_of_week: Tensor,
        weekly_weather: Tensor,
        weekly_hour: Tensor,
        weekly_day_of_week: Tensor,
    ) -> dict[str, Tensor]:
        if getattr(self.config, 'use_local_view', True):
            context = self.context_encoder(weather, hour_of_day, day_of_week)
            local = self.local_view(demand_history, context)
        else:
            local = demand_history.new_zeros(
                len(demand_history), self.local_view.num_active_nodes, self.config.d_model,
            )
        if self.periodic_mode == 'none':
            return {'local': local, 'fused': local}

        active = self.local_view.active_node_index(local.device)
        if active is not None:
            daily_demand = daily_demand.index_select(2, active)
            weekly_demand = weekly_demand.index_select(2, active)
        daily_context = weekly_context = None
        if self.periodic_mode == 'lstm':
            daily_context = self.context_encoder(daily_weather, daily_hour, daily_day_of_week)
            weekly_context = self.context_encoder(weekly_weather, weekly_hour, weekly_day_of_week)
        daily, daily_valid = self.daily_view(daily_demand, daily_mask, daily_context)
        weekly, weekly_valid = self.weekly_view(weekly_demand, weekly_mask, weekly_context)
        if not getattr(self.config, 'use_daily', True):
            daily_valid = torch.zeros_like(daily_valid)
        if not getattr(self.config, 'use_weekly', True):
            weekly_valid = torch.zeros_like(weekly_valid)
        local_mean = (
            demand_history.mean(dim=1).flatten(1)
            if self.periodic_mode == 'ma' and getattr(self.config, 'use_local_mean', True)
            else None
        )
        if local_mean is not None and active is not None:
            local_mean = local_mean.index_select(1, active)
        combined = self.fusion(
            local, daily, weekly, daily_valid, weekly_valid, active, local_mean,
        )
        views = {'local': local, 'daily': daily, 'weekly': weekly}
        if self.periodic_mode == 'lstm':
            views['fused'] = combined
        else:
            views['prediction'] = combined
        return views

    def forward_views(self, **batch) -> dict[str, Tensor]:
        """분석용 관점별 출력. 학습 경로에서는 추가 텐서를 반환하지 않는다."""
        views = self._encode_views(
            demand_history=batch['demand_history'],
            daily_demand=batch['daily_demand'],
            daily_mask=batch['daily_mask'],
            weekly_demand=batch['weekly_demand'],
            weekly_mask=batch['weekly_mask'],
            weather=batch['weather'],
            hour_of_day=batch['hour_of_day'],
            day_of_week=batch['day_of_week'],
            daily_weather=batch['daily_weather'],
            daily_hour=batch['daily_hour'],
            daily_day_of_week=batch['daily_day_of_week'],
            weekly_weather=batch['weekly_weather'],
            weekly_hour=batch['weekly_hour'],
            weekly_day_of_week=batch['weekly_day_of_week'],
        )
        views['logits'] = self._predict_grid(views)
        return views


__all__ = [
    'ContextEncoder',
    'FourierScalarEmbedding',
    'LocalViewEncoder',
    'MergedDemandModel',
    'PeriodicViewEncoder',
    'PredictionHead',
    'RetrievalViewEncoder',
    'ViewFusion',
    'WindowConvEncoder',
    'crop_local_windows',
    'make_neighbor_valid',
]
