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
        time_step: int = 24,
        temporal_n_layers: int = 2,
        temporal_n_heads: int = 4,
        temporal_dim_feedforward: int = 128,
        temporal_dropout: float = 0.1,
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
        self.time_step = time_step
        self.temporal_n_layers = temporal_n_layers
        self.temporal_n_heads = temporal_n_heads
        self.temporal_dim_feedforward = temporal_dim_feedforward
        self.temporal_dropout = temporal_dropout
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)
