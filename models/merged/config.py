from __future__ import annotations

from transformers import PretrainedConfig


class MergedDemandConfig(PretrainedConfig):
    """원본 ``UnifiedDemandModel.__init__``의 키워드 인자를 그대로 옮긴 HF 설정.

    ``GridDemandConfig``와 같은 패턴이다 — 파일 경로/데이터 파생값(``height``, ``width``,
    ``retrieval_grid_path``, ``retrieval_train_end``, ``weather_mean``, ``weather_std``)은
    학습 스크립트(``train_merged.py``)가 계산해서 주입하고, 나머지 하이퍼파라미터는
    ``configs/model/merged.yaml``에서 온다.

    9개 ablation 스위치(``use_daily``/``use_weekly``/``use_retrieval``/``use_weather``/
    ``weather_injection``/``use_calendar``/``use_branch_attention``/``use_neighbors``/
    ``use_softplus``)도 전부 여기 필드라 Hydra CLI 오버라이드로 켜고 끌 수 있다.
    """

    model_type = 'merged_demand'

    def __init__(
        self,
        height: int = 14,
        width: int = 12,
        time_step: int = 24,
        local_radius: int = 2,
        d_model: int = 64,
        num_fourier_bands: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 128,
        history_hidden: int = 64,
        periodic_hidden: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        retrieval_grid_path: str | None = None,
        retrieval_k: int = 20,
        retrieval_chunk_size: int = 256,
        retrieval_scope: str = 'observed_past',
        retrieval_train_end: int | None = None,
        weather_mean: list[float] | None = None,
        weather_std: list[float] | None = None,
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
        loss_type: str = 'combined',
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        **kwargs,
    ) -> None:
        self.height = height
        self.width = width
        self.time_step = time_step
        self.local_radius = local_radius
        self.d_model = d_model
        self.num_fourier_bands = num_fourier_bands
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ffn = transformer_ffn
        self.history_hidden = history_hidden
        self.periodic_hidden = periodic_hidden
        self.fusion_dim = fusion_dim
        self.dropout = dropout
        # 검색 브랜치가 읽는 temporal grid(.npy)의 절대경로. train_merged.py가 주입한다.
        self.retrieval_grid_path = retrieval_grid_path
        self.retrieval_k = retrieval_k
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieval_scope = retrieval_scope
        # 'train_prefix' 검색에서만 쓰이는 경계값. train split의 끝(절대 시간 인덱스).
        self.retrieval_train_end = retrieval_train_end
        # train split 통계(3개씩, leakage 방지 위해 train_merged.py가 계산해 주입).
        self.weather_mean = weather_mean
        self.weather_std = weather_std
        self.weekday_dim = weekday_dim
        self.hour_dim = hour_dim
        # --- ablation 스위치 ---
        self.use_daily = use_daily
        self.use_weekly = use_weekly
        self.use_retrieval = use_retrieval
        self.use_weather = use_weather
        self.weather_injection = weather_injection
        self.use_calendar = use_calendar
        self.use_branch_attention = use_branch_attention
        self.use_neighbors = use_neighbors
        self.use_softplus = use_softplus
        # 학습에 쓴 목적함수도 체크포인트에 남긴다 — from_pretrained로 되살렸을 때
        # loss 정의가 조용히 바뀌지 않도록.
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)


__all__ = ['MergedDemandConfig']
