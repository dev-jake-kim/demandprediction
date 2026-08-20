"""ST-ResNet(Zhang, Zheng, Qi. "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows
Prediction", AAAI 2017) 모델 포팅. 논문 Figure 3/4, 식(2)-(5)를 그대로 이식한다.
`docs/STRESNET_PLAN.md`에 논문 대비 우리가 내린 결정을 정리해뒀다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import STResNetConfig
from .losses import CombinedLoss

DAY_OF_WEEK_DIM = 7


class ResUnit(nn.Module):
    """Figure 4(b): pre-activation residual unit — (BN?+ReLU+Conv) x2 + identity."""

    def __init__(self, channels: int, use_bn: bool) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return x + h


class STResNetBranch(nn.Module):
    """closeness/period/trend 세 브랜치가 공유하는 구조 (식 2-3):
    Conv1 -> ResUnit x L -> ReLU -> Conv2."""

    def __init__(self, in_channels: int, out_channels: int, num_filters: int, num_res_units: int, use_bn: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.res_units = nn.ModuleList([ResUnit(num_filters, use_bn) for _ in range(num_res_units)])
        self.conv2 = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))  # 식(2): X^(1) = f(W^(1)*X^(0) + b^(1))
        for unit in self.res_units:
            h = unit(h)
        return self.conv2(F.relu(h))


class STResNetModel(PreTrainedModel):
    config_class = STResNetConfig
    base_model_prefix = 'stresnet'

    def __init__(self, config: STResNetConfig) -> None:
        super().__init__(config)
        H, W = config.H, config.W
        self.H, self.W = H, W

        # l_c/l_p/l_q=0이면 해당 브랜치를 아예 안 만듦(트렌드 등 특정 브랜치 비활성화 옵션 지원 —
        # 0채널 Conv2d를 만들어서 죽이는 대신 브랜치 자체를 생략).
        self.branch_c = (
            STResNetBranch(config.l_c, 1, config.num_filters, config.num_res_units, config.use_bn)
            if config.l_c > 0 else None
        )
        self.branch_p = (
            STResNetBranch(config.l_p, 1, config.num_filters, config.num_res_units, config.use_bn)
            if config.l_p > 0 else None
        )
        self.branch_q = (
            STResNetBranch(config.l_q, 1, config.num_filters, config.num_res_units, config.use_bn)
            if config.l_q > 0 else None
        )

        # 식(4): 파라미터 행렬 기반 fusion — 브랜치별 가중치를 위치(H,W)마다 따로 학습
        self.w_c = nn.Parameter(torch.ones(1, H, W)) if self.branch_c is not None else None
        self.w_p = nn.Parameter(torch.ones(1, H, W)) if self.branch_p is not None else None
        self.w_q = nn.Parameter(torch.ones(1, H, W)) if self.branch_q is not None else None

        # 외부 요인(day_of_week): 2-layer FC, 첫 층은 임베딩, 둘째 층은 X_t와 같은 shape으로 매핑
        self.ext_fc1 = nn.Linear(DAY_OF_WEEK_DIM, 10)
        self.ext_fc2 = nn.Linear(10, H * W)

        self.loss_fn = CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)

        self.post_init()

    def forward(
        self,
        demands_closeness: torch.Tensor | None = None,  # (B, l_c, H, W), l_c=0이면 안 옴
        demands_period: torch.Tensor | None = None,  # (B, l_p, H, W), l_p=0이면 안 옴
        demands_trend: torch.Tensor | None = None,  # (B, l_q, H, W), l_q=0이면 안 옴
        day_of_week: torch.Tensor | None = None,  # (B,)
        labels: torch.Tensor | None = None,  # (B, H, W)
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        H, W = self.H, self.W

        demand_min, demand_max = self.config.demand_min, self.config.demand_max
        denom = max(demand_max - demand_min, 1e-6)

        def normalize(x: torch.Tensor) -> torch.Tensor:
            # [-1, 1] Min-Max 정규화 (논문 preprocessing, tanh 출력 범위와 일치)
            return (x - demand_min) / denom * 2 - 1

        x_res = None
        for branch, weight, demands in (
            (self.branch_c, self.w_c, demands_closeness),
            (self.branch_p, self.w_p, demands_period),
            (self.branch_q, self.w_q, demands_trend),
        ):
            if branch is None:
                continue
            term = weight * branch(normalize(demands))  # (B,1,H,W), 식(4)의 항 하나
            x_res = term if x_res is None else x_res + term
        if x_res is None:
            raise ValueError("closeness/period/trend 브랜치가 전부 비활성화됨(l_c=l_p=l_q=0) — 최소 하나는 있어야 함")

        B = x_res.shape[0]
        day_onehot = F.one_hot(day_of_week, num_classes=DAY_OF_WEEK_DIM).to(x_res.dtype)
        ext = F.relu(self.ext_fc1(day_onehot))
        x_ext = self.ext_fc2(ext).reshape(B, 1, H, W)

        x_hat = torch.tanh(x_res + x_ext)  # 식(5), [-1,1] 정규화 공간

        pred_norm = x_hat.squeeze(1)  # (B,H,W)
        logits = (pred_norm + 1) / 2 * denom + demand_min  # 실제 수요 단위로 역정규화

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
