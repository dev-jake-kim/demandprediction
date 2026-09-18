from __future__ import annotations

from transformers import PretrainedConfig


class MergedDemandConfig(PretrainedConfig):
    """재설계된 수요 예측 모델의 설정.

    데이터에서 파생되는 값(``height``/``width``/``time_step``/``retrieval_grid_path``/
    ``retrieval_train_end``/``weather_mean``/``weather_std``)은 ``train.py``가 계산해 주입하고,
    나머지는 ``configs/model/merged_{ulsan,porto}.yaml``에서 온다.

    이전 설계에 있던 필드(transformer_*, history_hidden, periodic_hidden, fusion_dim,
    9개 ablation 스위치, node_adaptive*)는 전부 사라졌다. yaml에 아직 남아 있는 그 키들은
    ``PretrainedConfig``의 ``**kwargs``로 흘러들어가 보관만 되고 모델은 읽지 않는다.
    """

    model_type = 'merged_demand'

    def __init__(
        self,
        # --- 데이터 파생값 (train.py가 주입) ---
        height: int = 14,
        width: int = 12,
        time_step: int = 24,
        weather_mean: list[float] | None = None,
        weather_std: list[float] | None = None,
        retrieval_grid_path: str | None = None,
        retrieval_train_end: int | None = None,
        # --- 모델 폭 ---
        # 네 관점(local/daily/weekly/retrieval)이 전부 노드당 d_model 벡터 하나로 수렴한다.
        # 관점마다 폭을 따로 두지 않는 것이 이번 재설계의 핵심 단순화다.
        d_model: int = 64,
        dropout: float = 0.1,
        # --- local 관점 ---
        # 각 노드가 보는 이웃 창의 반지름 a. 창 크기는 (2a+1)^2.
        local_radius: int = 2,
        # 창 안의 스칼라 수요를 토큰으로 바꿀 때 쓰는 학습 가능한 Fourier 밴드 수.
        num_fourier_bands: int = 8,
        # 창 하나(길이 1+(2a+1)^2 시퀀스)를 요약하는 Transformer.
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
        weekday_dim: int = 7,
        hour_dim: int = 5,
        # --- 손실 ---
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
        self.weather_mean = weather_mean
        self.weather_std = weather_std
        self.retrieval_grid_path = retrieval_grid_path
        self.retrieval_train_end = retrieval_train_end

        self.d_model = d_model
        self.dropout = dropout
        self.local_radius = local_radius
        self.num_fourier_bands = num_fourier_bands
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ffn = transformer_ffn
        self.num_retrieval = num_retrieval
        self.retrieval_scope = retrieval_scope
        self.retrieval_chunk_size = retrieval_chunk_size
        self.weekday_dim = weekday_dim
        self.hour_dim = hour_dim

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
        """C = 정규화 날씨 3 + 요일 임베딩 + 시간대 임베딩."""
        return 3 + self.weekday_dim + self.hour_dim


__all__ = ['MergedDemandConfig']
