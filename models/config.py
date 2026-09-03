from __future__ import annotations

from transformers import PretrainedConfig


class ADFormerConfig(PretrainedConfig):
    """ADFormer(arXiv:2506.02576) 하이퍼파라미터.

    공식 구현(utils/ADFormer_config.py) 기본값을 따르되, N=H*W(168~200)에 맞춰
    cluster_reg_nums만 축소했다 (공식 기본값은 N=263 기준 [64,16]).
    """

    model_type = 'adformer'

    def __init__(
        self,
        H: int = 14,
        W: int = 12,
        time_step: int = 24,
        embed_dim: int = 64,
        SE_dim: int = 8,
        skip_dim: int = 256,
        s_heads: int = 3,
        sa_heads: int = 2,
        t_heads: int = 2,
        ta_heads: int = 1,
        cluster_seg_num: int = 2,
        depth: int = 6,
        mlp_ratio: int = 4,
        attn_drop: float = 0.0,
        agg_drop: float = 0.1,
        drop_path: float = 0.3,
        cluster_reg_nums: list[int] | None = None,
        cluster_map_path: str | None = None,
        demand_mean: float = 0.0,
        demand_std: float = 1.0,
        loss_type: str = 'combined',
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        **kwargs,
    ) -> None:
        self.H = H
        self.W = W
        self.time_step = time_step
        self.embed_dim = embed_dim
        self.SE_dim = SE_dim
        self.skip_dim = skip_dim
        self.s_heads = s_heads
        self.sa_heads = sa_heads
        self.t_heads = t_heads
        self.ta_heads = ta_heads
        self.cluster_seg_num = cluster_seg_num
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.agg_drop = agg_drop
        self.drop_path = drop_path
        self.cluster_reg_nums = cluster_reg_nums if cluster_reg_nums is not None else [40, 10]
        self.cluster_map_path = cluster_map_path
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        # 'combined'(기본, CombinedLoss=제곱오차+gamma*상대오차제곱) 또는 'mae'(L1).
        # 'mae'는 ADFormer 논문 공식 구현의 목적함수(raw 스케일 F.l1_loss)와 맞추기 위한
        # 옵션 — loss_gamma/loss_eps는 무시된다.
        self.loss_type = loss_type
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)
