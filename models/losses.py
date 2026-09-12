from __future__ import annotations

import math

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """절대오차 제곱 + gamma * 상대오차 제곱. y_true가 0에 가까울수록 상대오차 항이
    분모를 eps로 안정화한다."""

    def __init__(self, gamma: float = 1.0, eps: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        term1 = diff ** 2

        relative_diff = diff / (y_true + self.eps)
        term2 = relative_diff ** 2

        loss = term1 + self.gamma * term2

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class RmseMapeLoss(nn.Module):
    """``rmse_weight * RMSE + MAPE(+1)``.

    MAPE는 models/metrics.py의 보고 지표와 똑같이 분모 ``abs(y)+1`` 및 퍼센트 단위를 쓴다.
    rmse_weight=10이면 두 항이 (RMSE 0.75 -> 7.5) vs (MAPE 15) 정도로 비슷한 크기가 된다.

    학습 중 RMSE는 미니배치 단위 surrogate라 batch size에 영향을 받는다. validation의 best
    checkpoint 선택은 train.py가 전체 validation 예측으로 다시 계산한 동일 목적함수를 사용한다.

    **CombinedLoss와 달리 ``reduction`` 인자가 없다.** RMSE가 전체 원소에 걸친 하나의
    ``sqrt(mean(...))``이라 요소별 텐서로 분해되지 않기 때문이다. 항상 스칼라를 돌려준다 —
    ``models/merged/modeling.py``의 ``_compute``는 이 점을 알고 분기한다.
    """

    def __init__(self, rmse_weight: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(rmse_weight) or rmse_weight < 0:
            raise ValueError(f'rmse_weight는 0 이상이어야 함: got {rmse_weight}')
        self.rmse_weight = rmse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        # sqrt(mean(diff**2))와 값은 같지만, vector_norm은 diff가 전부 0일 때도
        # 0 gradient를 돌려준다. sqrt의 0 지점에서 생기는 0/0 gradient를 피하기 위함이다.
        rmse = torch.linalg.vector_norm(diff) / math.sqrt(diff.numel())
        mape = (diff.abs() / (y_true.abs() + 1.0)).mean() * 100.0
        return self.rmse_weight * rmse + mape


class DemandSplitLoss(nn.Module):
    """수요 구간별로 다른 오차를 쓴다 — 낮은 수요는 MAPE, 높은 수요는 제곱오차.

    ``rmse_weight * RMSE + MAPE(+1)``(:class:`RmseMapeLoss`)의 구조적 문제를 고치려고 만들었다.
    그 손실은 **모든 셀이 두 항 모두에 기여**하는데, 두 지표가 사실상 서로 다른 셀을 본다:

        ulsan test 73,416셀 (stage1 예측 기준)
          실제 수요 0인 셀: 전체의 75.1% — SSE 기여 7.9%,  MAPE 기여 39.6%
          실제 수요 2+ 셀 : 전체의 11.8% — SSE 기여 79.0%, MAPE 기여 32.0%

    MAPE(+1)의 분모가 ``|y|+1``이라 y=0이면 분모가 1이 되어 그냥 ``|오차|``가 되고, 0인 셀이
    압도적으로 많아 MAPE 전체를 좌우한다. 반대로 RMSE는 제곱이라 수요가 큰 소수 셀이 지배한다.
    그 결과 선형 결합은 "예측을 전반적으로 쪼그라뜨려 0셀 MAPE를 벌고 고수요 셀 RMSE를 잃는"
    거래를 남는 장사로 만든다(실측: RMSE +3.81%, MAPE(+1) -12.61%, 평균 예측 0.284 -> 0.233,
    실제 평균 0.467이므로 이미 과소예측인 상태에서 더 쪼그라듦).

    여기서는 각 셀을 **한 항에만** 넣어 그 교차 압력을 없앤다:

        y <= threshold : |오차| / (|y| + 1)      <- MAPE(+1)이 보는 것
        y >  threshold : high_weight * 오차^2    <- RMSE가 보는 것

    마스크가 ``target``에만 의존하므로 예측에 대한 gradient는 구간 안에서 연속이다.
    요소별 텐서를 그대로 돌려주므로(``reduction`` 인자가 없다) ``RmseMapeLoss``와 달리
    ``models/merged/modeling.py``의 스칼라 특수분기가 필요 없고 ``loss_sum`` 집계도 그대로 된다.

    ``high_weight``는 두 그룹의 균형을 잡는다. 기준은 **"관측된 나쁜 이동을 거부하는가"**다.
    ulsan의 stage1 -> stage2 이동(RMSE +3.81% / MAPE(+1) -12.61%, 평균예측 0.284 -> 0.233)은
    RMSE를 악화시키므로 채택 기준상 거부해야 하는 이동인데, 실제 예측으로 각 w를 재보면::

        w=0.117  Δ=-0.0178  보상함(나쁨)
        w=0.234  Δ=-0.0143  보상함(나쁨)    <- 두 그룹 loss 총량을 맞추는 값
        w=0.500  Δ=-0.0064  보상함(나쁨)
        w=1.000  Δ=+0.0086  거부함(좋음)    <- 기본값
        w=2.000  Δ=+0.0385  거부함(좋음)

    처음에는 "두 그룹의 loss 총량을 같게" 만드는 0.234를 기본값으로 잡았는데 틀렸다.
    중요한 건 총량이 아니라 **이동에 대한 민감도**다 — 고수요 셀은 전체의 12%뿐이라
    총량을 맞춰도 전역적으로 예측을 줄일 때의 반응이 저수요 쪽보다 둔하다. 전환점은
    w ~ 0.7~1.0 사이이고, 여유를 두어 1.0(가중치 없는 순수 제곱오차)을 기본값으로 쓴다.

    w를 키우면 고수요 셀(RMSE)로, 줄이면 저수요 셀(MAPE)로 무게가 옮겨간다.

    참고로 공용 :class:`CombinedLoss`(``오차^2 + gamma*(상대오차)^2``)는 이 이동을 애초에
    거부한다(Δ=+0.026). 제곱항이 고수요 셀의 과소예측을 2차로 벌주기 때문이다. 이 손실이
    갖는 차별점은 저수요 셀에서 제곱 대신 상대오차를 써서 MAPE(+1) 지표에 직접 맞춘다는
    점이고, 그 값어치는 실제로 돌려봐야 안다.
    """

    def __init__(self, threshold: float = 1.0, high_weight: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f'threshold는 0 이상이어야 함: got {threshold}')
        if not math.isfinite(high_weight) or high_weight < 0:
            raise ValueError(f'high_weight는 0 이상이어야 함: got {high_weight}')
        self.threshold = threshold
        self.high_weight = high_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        relative = diff.abs() / (y_true.abs() + 1.0)
        squared = diff ** 2
        is_low = y_true <= self.threshold
        return torch.where(is_low, relative, self.high_weight * squared)
