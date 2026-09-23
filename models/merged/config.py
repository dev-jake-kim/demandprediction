from __future__ import annotations

from transformers import PretrainedConfig


class MergedDemandConfig(PretrainedConfig):
    """재설계된 수요 예측 모델의 설정.

    데이터에서 파생되는 값(``height``/``width``/``time_step``/``retrieval_grid_path``/
    ``retrieval_train_end``/``weather_mean``/``weather_std``)은 ``train.py``가 계산해 주입하고,
    나머지는 ``configs/model/merged_{ulsan,porto}.yaml``에서 온다.

    ``node_adaptive`` / ``node_adaptive_min_demand`` / ``node_adaptive_indices``는
    train.py가 train 구간 통계로 결정해 주입하는 노드별 temporal LSTM 적응 설정이다.
    ``shared_weight_fp8``은 공유 LSTM weight에만 fake-FP8 양자화를 적용하는 실험 스위치다.
    yaml에 남아 있는 다른 키들은 ``PretrainedConfig``의 ``**kwargs``로 흘러들어가 보관된다.

    """
    model_type = 'merged_demand'

    def __init__(
        self,
        # --- 데이터 파생값 (train.py가 주입) ---
        height: int = 14,
        width: int = 12,
        time_step: int = 24,
        # train.py가 계산해 주입하지만 모델은 더 이상 쓰지 않는다 — 어떤 통계로 돌린
        # 런인지 체크포인트에 남기는 기록용이다. 정규화는 아래 min/max로 한다.
        weather_mean: list[float] | None = None,
        weather_std: list[float] | None = None,
        retrieval_grid_path: str | None = None,
        retrieval_train_end: int | None = None,
        # --- 모델 폭 ---
        # 네 관점(local/daily/weekly/retrieval)이 전부 노드당 d_model 벡터 하나로 수렴한다.
        # 관점마다 폭을 따로 두지 않는 것이 이번 재설계의 핵심 단순화다.
        d_model: int = 64,
        dropout: float = 0.1,
        # ①어텐션 가중치 dropout만 따로 뗀 값. nn.TransformerEncoderLayer는 dropout 하나를
        # 네 곳(어텐션 가중치 / 어텐션 출력 / FFN 은닉 / FFN 출력)에 모두 쓰는데, 앞의 하나만
        # 성격이 다르다 — 활성값의 원소를 끄는 게 아니라 "CLS가 이 이웃 칸을 보는 연결"을
        # 통째로 끊는다. 수요 격자는 셀의 74%가 0이라 정보를 가진 이웃이 몇 개 안 되고,
        # 그중 하나를 끊는 것은 상대적으로 매우 큰 교란이다. 기본값 0.0으로 끈다.
        # 부수효과: PyTorch의 SDPA는 batch > 65535일 때 어텐션 dropout이 0이 아니면
        # 거부한다(attention.cu). 이 값이 0이면 그 한계가 사라진다.
        attention_dropout: float = 0.0,
        # --- local 관점 ---
        # 각 노드가 보는 이웃 창의 반지름 a. 창 크기는 (2a+1)^2.
        local_radius: int = 2,
        # 창 안의 스칼라 수요를 토큰으로 바꿀 때 쓰는 학습 가능한 Fourier 밴드 수.
        num_fourier_bands: int = 8,
        # 창 인코더 종류. 'transformer' = CLS + 이웃 토큰 self-attention(기본),
        # 'conv' = 창을 (2a+1)x(2a+1) 이미지로 보고 residual 3x3 conv 블록을 쌓는다
        # (위치 임베딩 없음, 요약은 중앙 칸). 두 경로 모두 transformer_layers만큼 쌓는다.
        local_encoder: str = 'transformer',
        # 창 하나를 요약하는 블록 수.
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 128,
        # --- retrieval 관점 ---
        # 최근 k시간 패턴과 비슷한 과거 시점을 몇 개 가져올지.
        num_retrieval: int = 20,
        # 후보 범위. 'observed_past' = tau < target_time, 'train_prefix' = train split까지만.
        retrieval_scope: str = 'observed_past',
        retrieval_chunk_size: int = 256,
        # --- 보조 정보(날씨/캘린더) ---
        # 날씨 2채널의 min-max 정규화 기준. **train 구간에서 재서 여기 적어 넣는다**
        # (configs/model/merged_<city>.yaml). 시간 리크를 막으려면 val/test를 보면 안 된다.
        temperature_min: float | None = None,
        temperature_max: float | None = None,
        # (강수량 + snow_scale * 적설)의 train 구간 최대값. snow_scale은 학습 중 계속
        # 변하므로 그때그때 다시 잴 수 없다 — **snow_scale=1.0 기준**으로 한 번 재서 고정한다.
        # 최소값은 0이다(두 값 모두 음수가 될 수 없다).
        precipitation_max: float | None = None,
        weekday_dim: int = 7,
        hour_dim: int = 5,
        # --- 노드별 temporal LSTM weight offset ---
        # node_adaptive=true이면 train.py가 train 구간 평균 수요 기준으로 고른
        # node_adaptive_indices에만 fp32 ΔW를 만든다. 목록은 버퍼가 아니라 config의
        # 파이썬 리스트로 보관한다(from_pretrained meta 초기화에서의 쓰레기값 방지).
        node_adaptive: bool = False,
        node_adaptive_min_demand: float = 0.8,
        node_adaptive_indices: list[int] | None = None,
        # 공유 LSTM weight 4개를 absmax-scaled torch.float8_e4m3fn으로 fake quantize한다.
        # ΔW와 결합 스칼라 s는 항상 fp32로 유지한다.
        shared_weight_fp8: bool = False,
        # --- 항상 0인 노드 제외 ---
        # train 구간 평균 수요가 이 값 이하인 노드는 학습에서 뺀다: 예측을 정확히 0으로
        # 고정하고 손실에서도 제외해, 남은 노드에만 용량과 gradient가 가도록 한다.
        # None이면 이 기능이 없던 때와 완전히 동일하게 동작한다.
        zero_node_max_demand: float | None = None,
        # 위 기준으로 실제 선택된 노드 id. 시간 리크를 막으려면 train 구간에서만 계산해야
        # 하므로 train.py가 계산해 주입한다. 체크포인트에 남겨야 같은 마스크가 복원된다.
        # **버퍼로 만들면 안 된다** — from_pretrained의 meta device 초기화에서 체크포인트에
        # 없는 버퍼는 torch.empty(쓰레기값)로 남는다.
        zero_node_indices: list[int] | None = None,
        # --- 손실 ---
        loss_type: str = 'mae',
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        rmse_weight: float = 10.0,
        split_threshold: float = 1.0,
        split_high_weight: float = 1.0,
        **kwargs,
    ) -> None:
        self.height = height
        self.width = width
        self.time_step = time_step
        self.weather_mean = weather_mean
        self.weather_std = weather_std
        self.retrieval_grid_path = retrieval_grid_path
        self.retrieval_train_end = retrieval_train_end

        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.local_radius = local_radius
        self.num_fourier_bands = num_fourier_bands
        self.local_encoder = str(local_encoder)
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ffn = transformer_ffn
        self.num_retrieval = num_retrieval
        self.retrieval_scope = retrieval_scope
        self.retrieval_chunk_size = retrieval_chunk_size
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.precipitation_max = precipitation_max
        self.weekday_dim = weekday_dim
        self.hour_dim = hour_dim

        self.node_adaptive = bool(node_adaptive)
        self.node_adaptive_min_demand = float(node_adaptive_min_demand)
        self.node_adaptive_indices = node_adaptive_indices
        self.shared_weight_fp8 = bool(shared_weight_fp8)
        self.zero_node_max_demand = zero_node_max_demand
        if self.node_adaptive and (
            self.zero_node_max_demand is not None or zero_node_indices
        ):
            raise ValueError(
                'node_adaptive=true는 zero_node_max_demand/zero_node_indices와 '
                '동시에 사용할 수 없음: 적응 노드 id와 잘린 노드 축이 어긋남'
            )
        self.zero_node_indices = zero_node_indices
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        self.rmse_weight = rmse_weight
        self.split_threshold = split_threshold
        self.split_high_weight = split_high_weight
        super().__init__(**kwargs)

    # 파생값 — 모델 곳곳에서 쓰이는 축 길이를 한 곳에서만 정의한다.
    @property
    def num_nodes(self) -> int:
        """N = H * W."""
        return self.height * self.width

    @property
    def window_size(self) -> int:
        """2a + 1."""
        return 2 * self.local_radius + 1

    @property
    def num_neighbors(self) -> int:
        """P = (2a+1)^2. 창 하나의 토큰 수는 CLS를 포함해 1 + P."""
        return self.window_size * self.window_size

    @property
    def context_dim(self) -> int:
        """C = 날씨 2채널(기온, 총강수) + 요일 임베딩 + 시간대 임베딩.

        CSV는 기온/강수량/적설 3개지만 ``ContextEncoder``가 강수량과 적설을 학습 가능한
        환산계수로 하나의 총강수 채널로 합친다(:class:`~models.merged.modeling.ContextEncoder`).
        """
        return 2 + self.weekday_dim + self.hour_dim


__all__ = ['MergedDemandConfig']
