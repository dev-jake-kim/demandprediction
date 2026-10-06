from __future__ import annotations

from transformers import PretrainedConfig


class MergedDemandConfig(PretrainedConfig):
    """Configuration for the merged demand forecasting model.

    Data-derived values such as grid dimensions, train-split weather extremes, and retrieval
    bounds are supplied by the training pipeline; the remaining fields are model settings.
    """

    model_type = 'merged_demand'

    def __init__(
        self,
        height: int = 14,
        width: int = 12,
        time_step: int = 24,
        local_radius: int = 2,
        retrieval_local_radius: int | None = None,
        d_model: int = 64,
        num_fourier_bands: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 128,
        history_hidden: int = 64,
        periodic_hidden: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float | None = None,
        retrieval_grid_path: str | None = None,
        retrieval_k: int = 20,
        retrieval_chunk_size: int = 256,
        retrieval_scope: str = 'observed_past',
        retrieval_train_end: int | None = None,
        context_extra_dim: int = 0,
        context_features: list[str] | None = None,
        functional_neighbor_k: int = 0,
        functional_neighbors: list[list[int]] | None = None,
        functional_neighbor_mode: str = 'raw',
        temperature_min: float | None = None,
        temperature_max: float | None = None,
        precipitation_max: float | None = None,
        weekday_dim: int = 7,
        hour_dim: int = 5,
        use_daily: bool = True,
        use_weekly: bool = True,
        use_retrieval: bool = True,
        use_weather: bool = True,
        weather_injection: str = 'concat',
        use_calendar: bool = True,
        use_branch_attention: bool = True,
        use_neighbors: bool = True,
        use_sink: bool = True,
        local_encoder: str = 'masked_grid',
        global_nodes: list[int] | None = None,
        aux_channels: int = 0,
        aux_grid_path: str | None = None,
        use_softplus: bool = True,
        node_adaptive: bool = False,
        node_adaptive_min_demand: float = 0.8,
        node_adaptive_indices: list[int] | None = None,
        use_event_embedding: bool = False,
        event_csv_path: str | None = None,
        weather_lead: int = 0,
        node_adaptive_mode: str = 'additive',
        loss_type: str = 'combined',
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        rmse_weight: float = 10.0,
        split_threshold: float = 1.0,
        split_high_weight: float = 1.0,
        mape_weight: float = 1.0,
        node_mix_lambda_low: float = 0.9,
        node_mix_lambda_high: float = 0.1,
        node_mix_tau: float = 0.5,
        node_mix_width: float = 0.5,
        node_mix_scaler: bool = True,
        node_mix_lambda: list[float] | None = None,
        node_mix_mape_scale: list[float] | None = None,
        node_mix_mse_scale: list[float] | None = None,
        node_mix_weight: list[float] | None = None,
        **kwargs,
    ) -> None:
        self.height = height
        self.width = width
        self.time_step = time_step
        self.local_radius = local_radius
        self.retrieval_local_radius = retrieval_local_radius
        self.d_model = d_model
        self.num_fourier_bands = num_fourier_bands
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ffn = transformer_ffn
        self.history_hidden = history_hidden
        self.periodic_hidden = periodic_hidden
        self.fusion_dim = fusion_dim
        self.dropout = dropout
        self.attention_dropout = dropout if attention_dropout is None else float(attention_dropout)
        self.retrieval_grid_path = retrieval_grid_path
        self.retrieval_k = retrieval_k
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieval_scope = retrieval_scope
        self.retrieval_train_end = retrieval_train_end
        # 데이터셋 추가 context 채널(train.py가 data.context_features에서 주입, test.py가 재구성에 씀).
        self.context_extra_dim = context_extra_dim
        self.context_features = list(context_features or [])
        # 기능적 이웃(train.py가 train 구간 상관으로 골라 주입). k=0이면 끔.
        self.functional_neighbor_k = functional_neighbor_k
        self.functional_neighbors = functional_neighbors
        self.functional_neighbor_mode = functional_neighbor_mode
        # train 구간 통계. precipitation_max는 snow_scale=1 기준으로 고정한다.
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.precipitation_max = precipitation_max
        self.weekday_dim = weekday_dim
        self.hour_dim = hour_dim
        self.use_daily = use_daily
        self.use_weekly = use_weekly
        self.use_retrieval = use_retrieval
        self.use_weather = use_weather
        self.weather_injection = weather_injection
        self.use_calendar = use_calendar
        self.use_branch_attention = use_branch_attention
        self.use_neighbors = use_neighbors
        self.use_sink = use_sink
        # masked_grid(채택) | crop(이전 방식, ablation 전용) — docs/SPEC.md §4.2
        self.local_encoder = local_encoder
        # 모든 노드가 attention으로 볼 수 있는 노드 id(마스크 열 개방, docs/SPEC.md §6.2).
        self.global_nodes = list(global_nodes or [])
        # 노드별 배차 상태 채널 수와 격자 경로(train.py가 data.aux_grid에서 주입, docs/SPEC.md §4).
        self.aux_channels = aux_channels
        self.aux_grid_path = aux_grid_path
        self.use_softplus = use_softplus
        self.node_adaptive = node_adaptive
        self.node_adaptive_min_demand = node_adaptive_min_demand
        # train.py가 train 구간에서 계산해 주입하고, 체크포인트 복원에 쓰인다.
        self.node_adaptive_indices = node_adaptive_indices
        # 공연·축제 등 이벤트 노드·시각에 공유 학습 임베딩을 더한다. 표 경로는 train.py가 절대경로로 주입.
        self.use_event_embedding = use_event_embedding
        self.event_csv_path = event_csv_path
        # 최근 창 날씨 행을 당기는 칸 수(데이터셋 설정). 모델 연산에는 쓰지 않고 test.py 재현용으로 기록.
        self.weather_lead = weather_lead
        # additive: W + ΔW[n] | gated: W + s·(1−g[n])·(V[n] − W) (docs/SPEC.md §6.7)
        self.node_adaptive_mode = node_adaptive_mode
        # configure_loss가 stage마다 갱신한다(체크포인트가 마지막 학습 손실을 기록).
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        self.rmse_weight = rmse_weight
        self.split_threshold = split_threshold
        self.split_high_weight = split_high_weight
        # mse_mape 손실의 상대오차 항 가중치(error² + mape_weight·|error|/(y+1)).
        self.mape_weight = mape_weight
        # node_mix 손실 하이퍼파라미터와 train.py가 계산해 넣는 노드별 벡터(docs/SPEC.md §7).
        self.node_mix_lambda_low = node_mix_lambda_low
        self.node_mix_lambda_high = node_mix_lambda_high
        self.node_mix_tau = node_mix_tau
        self.node_mix_width = node_mix_width
        self.node_mix_scaler = node_mix_scaler
        self.node_mix_lambda = node_mix_lambda
        self.node_mix_mape_scale = node_mix_mape_scale
        self.node_mix_mse_scale = node_mix_mse_scale
        self.node_mix_weight = node_mix_weight
        super().__init__(**kwargs)


__all__ = ['MergedDemandConfig']
