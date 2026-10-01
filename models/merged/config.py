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
        retrieval_train_end: int | None = None,
        retrieval_future_mask_hours: int = 72,
        retrieval_value_cap: int | None = None,
        retrieval_embedding_dim: int = 16,
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
        use_softplus: bool = True,
        node_adaptive: bool = False,
        node_adaptive_min_demand: float = 0.8,
        node_adaptive_indices: list[int] | None = None,
        loss_type: str = 'combined',
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
        self.retrieval_train_end = retrieval_train_end
        # train 모드에서 [t, t + 이 값] 후보를 가린다. time_step 이상이어야 누수가 없다.
        self.retrieval_future_mask_hours = retrieval_future_mask_hours
        # train 구간 상위 0.5% 수요 경계(train.py가 주입). 이 값 이상은 한 embedding bucket이다.
        self.retrieval_value_cap = retrieval_value_cap
        self.retrieval_embedding_dim = retrieval_embedding_dim
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
        self.use_softplus = use_softplus
        self.node_adaptive = node_adaptive
        self.node_adaptive_min_demand = node_adaptive_min_demand
        # train.py가 train 구간에서 계산해 주입하고, 체크포인트 복원에 쓰인다.
        self.node_adaptive_indices = node_adaptive_indices
        # configure_loss가 stage마다 갱신한다(체크포인트가 마지막 학습 손실을 기록).
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        self.rmse_weight = rmse_weight
        self.split_threshold = split_threshold
        self.split_high_weight = split_high_weight
        super().__init__(**kwargs)


__all__ = ['MergedDemandConfig']
