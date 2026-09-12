from __future__ import annotations

from transformers import PretrainedConfig


class MergedDemandConfig(PretrainedConfig):
    """원본 ``UnifiedDemandModel.__init__``의 키워드 인자를 그대로 옮긴 HF 설정.

    ``GridDemandConfig``와 같은 패턴이다 — 파일 경로/데이터 파생값(``height``, ``width``,
    ``retrieval_grid_path``, ``retrieval_train_end``, ``weather_mean``, ``weather_std``)은
    학습 스크립트(``train.py``)가 계산해서 주입하고, 나머지 하이퍼파라미터는 도시별
    ``configs/model/merged_ulsan.yaml`` / ``configs/model/merged_porto.yaml``에서 온다.

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
        self.d_model = d_model
        self.num_fourier_bands = num_fourier_bands
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ffn = transformer_ffn
        self.history_hidden = history_hidden
        self.periodic_hidden = periodic_hidden
        self.fusion_dim = fusion_dim
        self.dropout = dropout
        # 검색 브랜치가 읽는 temporal grid(.npy)의 절대경로. train.py가 주입한다.
        self.retrieval_grid_path = retrieval_grid_path
        self.retrieval_k = retrieval_k
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieval_scope = retrieval_scope
        # 'train_prefix' 검색에서만 쓰이는 경계값. train split의 끝(절대 시간 인덱스).
        self.retrieval_train_end = retrieval_train_end
        # train split 통계(3개씩, leakage 방지 위해 train.py가 계산해 주입).
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
        # --- 노드별 LSTM weight offset (lora 브랜치의 node_adaptive와 같은 구조) ---
        # history LSTM의 weight/bias를 노드마다 다르게 쓴다(공유 W + node id별 ΔW, 0으로 초기화).
        # False면 이 필드가 없던 시절과 완전히 동일하게 동작한다 — 기본값을 False로 두어야
        # tests/test_merged_parity.py(포팅 직전 구현과 대조)와 기존 run JSON 비교가 유지된다.
        # daily/weekly 주기 브랜치는 대상이 아니다(pack_padded_sequence가 노드 축을 흐트러뜨림).
        self.node_adaptive = node_adaptive
        # ΔW를 받을 노드를 고르는 기준: train 구간 평균 수요가 이 값을 넘는 노드만.
        # 수요가 거의 0인 노드(porto는 노드 중앙값이 0.009다)에 노드당 수만 개의 파라미터를
        # 주면 신호가 아니라 노이즈를 외울 용량만 늘어난다. 나머지 노드는 공유 W만 쓰며
        # cuDNN fused LSTM 경로를 그대로 타서 속도 손해도 없다.
        self.node_adaptive_min_demand = node_adaptive_min_demand
        # 위 기준으로 실제 선택된 노드 id(0..H*W-1) 목록. 시간 리크를 막으려면 train 구간에서만
        # 계산해야 하므로 weather_mean/std와 같이 train.py가 계산해 주입한다. 체크포인트에
        # 남겨야 from_pretrained가 같은 마스크를 복원한다.
        self.node_adaptive_indices = node_adaptive_indices
        # 학습에 쓴 목적함수도 체크포인트에 남긴다 — from_pretrained로 되살렸을 때
        # loss 정의가 조용히 바뀌지 않도록. 2-stage 학습은 stage마다 loss를 갈아끼우므로
        # configure_loss()가 이 필드를 함께 갱신한다.
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        # loss_type='rmse_mape'에서만 쓰인다: rmse_weight * RMSE + MAPE(+1).
        self.rmse_weight = rmse_weight
        # loss_type='demand_split'에서만 쓰인다. 실제 수요가 split_threshold 이하인 셀은
        # MAPE(+1) 상대오차로, 초과인 셀은 split_high_weight * 제곱오차로 벌준다.
        self.split_threshold = split_threshold
        self.split_high_weight = split_high_weight
        super().__init__(**kwargs)


__all__ = ['MergedDemandConfig']
