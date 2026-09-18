"""재설계된 수요 예측 모델 — 골격.

구현이 끝난 것은 :class:`LocalViewEncoder` 하나이고 나머지는 ``forward``가 ``pass``다.
비어 있는 블록이 확정하는 것은 **정보의 흐름**, 즉 어떤 정보를 받아 어떤 shape로 내보내는지뿐이다.

거시적 구조::

    demand_history [B,k,H,W] ─┬─> LocalViewEncoder      ──> [B,N,D]  "내 주변에서 최근 k시간"
                              │
                              └─> RetrievalViewEncoder  ──> [B,N,D]  "과거의 비슷했던 n개 시점"
    daily_demand   [B,Ld,N,1] ──> PeriodicViewEncoder   ──> [B,N,D]  "며칠 전 같은 시간대"
    weekly_demand  [B,Lw,N,1] ──> PeriodicViewEncoder   ──> [B,N,D]  "몇 주 전 같은 요일·시간대"

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

    날씨는 임베딩하지 않고 train split 통계로 정규화한 3값을 그대로 쓴다.
    정규화 후의 0은 "평균 날씨" = 정보 없음의 자연스러운 대체값이다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()

    def forward(self, weather: Tensor, hour_of_day: Tensor, day_of_week: Tensor) -> Tensor:
        """
        Args:
            weather: ``[B, L, 3]`` 원본 스케일 (기온, 강수량, 적설).
            hour_of_day: ``[B, L]`` 0..23 정수.
            day_of_week: ``[B, L]`` 0..6 정수.

        Returns:
            ``[B, L, C]`` 시점별 보조 정보.
        """

        pass


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
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()
        if config.d_model % config.transformer_heads != 0:
            raise ValueError('d_model은 transformer_heads로 나누어떨어져야 함')

        d_model = config.d_model
        self.local_radius = config.local_radius
        self.num_nodes = config.num_nodes
        self.num_neighbors = config.num_neighbors

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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        # 시간 축을 접는다. 입력은 시점별 창 요약(D) + 그 시점의 보조 정보(C).
        self.temporal_lstm = nn.LSTM(d_model + config.context_dim, d_model, batch_first=True)

    def forward(self, demand_history: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            demand_history: ``[B, k, H, W]`` 최근 k시간의 격자 전체 수요(raw 스케일).
            context: ``[B, k, C]`` 같은 k시점의 보조 정보.

        Returns:
            ``[B, N, D]`` 노드별 공간 관점 임베딩.
        """

        local_crop = crop_local_windows(demand_history, self.local_radius)
        batch, steps, nodes, neighbors = local_crop.shape

        # 수요는 꼬리가 긴 분포라 log1p로 압축한 뒤 토큰으로 만든다.
        values = torch.log1p(torch.clamp(local_crop, min=0.0)).unsqueeze(-1)
        tokens = self.scalar_embedding(values)  # [B,k,N,P,D]
        # 격자 밖은 "수요 0"이 아니라 "값이 없음"이다 — 전용 EDGE 토큰으로 갈아끼운다.
        edge_token = self.special_embedding.weight[1].view(1, 1, 1, 1, -1)
        tokens = torch.where(
            self.neighbor_valid.view(1, 1, nodes, neighbors, 1), tokens, edge_token
        )

        cls_token = self.special_embedding.weight[0].view(1, 1, 1, 1, -1)
        cls_token = cls_token + self.node_embedding.view(1, 1, nodes, 1, -1)
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
        _, (hidden, _) = self.temporal_lstm(sequence)
        return hidden[-1].reshape(batch, nodes, -1)


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
    """네 관점을 노드별로 하나의 벡터로 합친다.

    합치는 방식(가중 평균 / 어텐션 / concat+MLP)은 아직 정하지 않았다. 확정된 것은
    입출력 규약뿐이라, 관점 수 ``V``가 바뀌어도 이 블록의 시그니처는 그대로다.
    """

    def __init__(self, config: MergedDemandConfig, num_views: int) -> None:
        super().__init__()
        self.num_views = num_views

    def forward(self, views: Tensor) -> Tensor:
        """
        Args:
            views: ``[B, N, V, D]`` 관점별 노드 임베딩.
                V=4, 순서는 (local, daily, weekly, retrieval).

        Returns:
            ``[B, N, D]`` 융합된 노드 임베딩.
        """

        pass


class PredictionHead(nn.Module):
    """융합된 노드 임베딩 -> 그 노드의 다음 1시간 수요(스칼라).

    수요는 음수가 될 수 없으므로 마지막에 비음수 활성(Softplus 등)을 씌운다.
    """

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__()

    def forward(self, fused: Tensor) -> Tensor:
        """
        Args:
            fused: ``[B, N, D]``

        Returns:
            ``[B, N]`` 노드별 예측 수요(raw 스케일, >= 0).
        """

        pass


class MergedDemandModel(PreTrainedModel):
    """네 관점(local / daily / weekly / retrieval)을 융합하는 수요 예측 모델."""

    config_class = MergedDemandConfig
    base_model_prefix = 'merged_demand'
    main_input_name = 'demand_history'

    # 융합에 들어가는 순서. ViewFusion의 V 축 의미를 여기 한 곳에서만 정의한다.
    VIEW_NAMES = ('local', 'daily', 'weekly', 'retrieval')

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__(config)

        self.context_encoder = ContextEncoder(config)
        self.local_view = LocalViewEncoder(config)
        # daily/weekly는 같은 클래스의 별개 인스턴스 — 파라미터를 공유하지 않는다.
        self.daily_view = PeriodicViewEncoder(config)
        self.weekly_view = PeriodicViewEncoder(config)
        self.retrieval_view = RetrievalViewEncoder(config)
        self.fusion = ViewFusion(config, num_views=len(self.VIEW_NAMES))
        self.head = PredictionHead(config)

        self.loss_fn: nn.Module
        self.configure_loss(config.loss_type)
        self.post_init()

    def configure_loss(self, loss_type: str, *, rmse_weight: float | None = None) -> None:
        """학습 loss를 교체하고 선택값을 config에도 동기화한다.

        config를 함께 갱신해야 체크포인트를 ``from_pretrained``로 다시 열었을 때 런타임에
        골랐던 loss가 조용히 되돌아가지 않는다.
        """

        if rmse_weight is not None:
            self.config.rmse_weight = float(rmse_weight)
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

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        """노드별 weight offset(ΔW). 재설계에서 빠졌으므로 항상 비어 있다.

        ``train.py``의 2-stage 학습이 이 메서드를 부르기 때문에 스텁으로 남긴다 —
        노드별 적응을 다시 넣을지는 아직 정하지 않았다.
        """

        return ()

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
        """

        pass

    def forward_views(self, **batch) -> dict[str, Tensor]:
        """관점별 임베딩을 그대로 돌려준다(분석/디버깅용, 학습 경로에서 쓰지 않는다).

        Returns:
            ``{'local'|'daily'|'weekly'|'retrieval': [B, N, D], 'fused': [B, N, D],
            'logits': [B, H, W]}``
        """

        pass


__all__ = [
    'ContextEncoder',
    'FourierScalarEmbedding',
    'LocalViewEncoder',
    'MergedDemandModel',
    'PeriodicViewEncoder',
    'PredictionHead',
    'RetrievalViewEncoder',
    'ViewFusion',
    'crop_local_windows',
    'make_neighbor_valid',
]
