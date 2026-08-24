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
        commute_map_path: str | None = None,
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
        # commute-attention 전용 — preprocessing/build_commute_map.py가 만든 (N,n) 참조 인덱스/유사도
        # npz의 절대경로(train.py가 주입). n은 여기서 별도로 안 두고 로드된 배열의 shape에서 읽는다.
        self.commute_map_path = commute_map_path
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)
