from __future__ import annotations

from transformers import PretrainedConfig


class STResNetConfig(PretrainedConfig):
    """ST-ResNet(Zhang et al., AAAI 2017) 하이퍼파라미터.

    논문 최고 성능 변형(L12-E-BN: residual unit 12개, BatchNorm 포함, 외부 요인 사용)을 기본값으로
    한다. `l_c/l_p/l_q`는 우리 데이터(ulsan 182일, porto 365일)가 논문 실험(1년+)보다 짧아
    trend lookback으로 인한 학습 샘플 손실을 줄이기 위해 작게 잡았다.
    """

    model_type = 'stresnet'

    def __init__(
        self,
        H: int = 14,
        W: int = 12,
        l_c: int = 3,
        l_p: int = 1,
        l_q: int = 1,
        num_filters: int = 64,
        num_res_units: int = 12,
        use_bn: bool = True,
        demand_min: float = 0.0,
        demand_max: float = 1.0,
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
        **kwargs,
    ) -> None:
        self.H = H
        self.W = W
        self.l_c = l_c
        self.l_p = l_p
        self.l_q = l_q
        self.num_filters = num_filters
        self.num_res_units = num_res_units
        self.use_bn = use_bn
        self.demand_min = demand_min
        self.demand_max = demand_max
        self.loss_gamma = loss_gamma
        self.loss_eps = loss_eps
        super().__init__(**kwargs)
