from __future__ import annotations

from transformers import PretrainedConfig


class DMVSTConfig(PretrainedConfig):
    """DMVST-Net(Yao et al., AAAI 2018) 하이퍼파라미터.

    `patch_size`(=S, 논문의 로컬 이웃 크기)는 `master`의 `GridDemandModel`이 쓰던
    `a`(반경) 대신 지름으로 받는다 — `a = (patch_size - 1) // 2`로 내부에서 변환해
    이웃 테이블(`idx_table`/`mask_table`)을 만든다.
    """

    model_type = 'dmvst'

    def __init__(
        self,
        H: int = 14,
        W: int = 12,
        time_step: int = 8,
        patch_size: int = 9,
        num_filters: int = 4,
        num_cnn_layers: int = 3,
        kernel_size: int = 3,
        demand_embedding_dim: int = 16,
        temporal_embedding_dim: int = 8,
        context_embedding_dim: int = 8,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        line_dim: int = 64,
        line_embeddings_path: str | None = None,
        demand_min: float = 0.0,
        demand_max: float = 1.0,
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        **kwargs,
    ) -> None:
        self.H = H
        self.W = W
        self.time_step = time_step
        self.patch_size = patch_size
        self.num_filters = num_filters
        self.num_cnn_layers = num_cnn_layers
        self.kernel_size = kernel_size
        self.demand_embedding_dim = demand_embedding_dim
        self.temporal_embedding_dim = temporal_embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.line_dim = line_dim
        self.line_embeddings_path = line_embeddings_path
        self.demand_min = demand_min
        self.demand_max = demand_max
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)
