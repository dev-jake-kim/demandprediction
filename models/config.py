from __future__ import annotations

from transformers import PretrainedConfig


class GridDemandConfig(PretrainedConfig):
    model_type = 'grid_demand'

    def __init__(
        self,
        H: int = 14,
        W: int = 12,
        a: int = 2,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        time_step: int = 24,
        npy_path: str | None = None,
        retrieval_k: int = 20,
        weather_csv_path: str | None = None,
        weather_mean: list[float] | None = None,
        weather_std: list[float] | None = None,
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        **kwargs,
    ) -> None:
        self.H = H
        self.W = W
        self.a = a
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        # 아래 3개는 검색(retrieval) 브랜치 전용 — time_step/npy_path는 검색 DB(build_retrieval_db)를
        # 만드는 데 필요하고, npy_path는 원본 grid 전체 시계열을 가리키는 절대경로여야 함(train.py가 주입).
        self.time_step = time_step
        self.npy_path = npy_path
        self.retrieval_k = retrieval_k
        # 날씨/캘린더 브랜치 전용 — weather_csv_path는 절대경로(train.py가 주입),
        # weather_mean/std는 train split 통계(3개씩, leakage 방지 위해 train.py가 계산해 주입).
        self.weather_csv_path = weather_csv_path
        self.weather_mean = weather_mean
        self.weather_std = weather_std
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)
