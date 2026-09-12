"""Training objectives for the merged model.

원본 ``merged_model/losses.py``는 저장소 공용 ``CombinedLoss``를 파일 안에 복사해 두고
있었다. 여기서는 복사본 대신 ``models/losses.py``의 원본을 그대로 재사용한다 —
``reduction='none'``이면 두 구현의 반환값이 완전히 동일하다(요소별 텐서).

모든 손실은 ``reduction='none'``이다. 모델이 한 번만 reduce하므로 epoch 집계가
배치 평균의 평균이 아니라 진짜 요소 평균이 된다(``drop_last=False``라 배치 크기가 다르다).
"""

from __future__ import annotations

from torch import nn

from ..losses import CombinedLoss, DemandSplitLoss, RmseMapeLoss

# 요소별(reduction='none') 텐서를 돌려주지 않고 스칼라 하나를 돌려주는 손실.
# modeling.py의 _compute가 mean()/sum() 집계를 건너뛰어야 하는지 판단할 때 쓴다.
SCALAR_LOSS_TYPES = frozenset({'rmse_mape'})


def build_loss(
    loss_type: str,
    *,
    gamma: float = 1.0,
    eps: float = 0.5,
    rmse_weight: float = 10.0,
    split_threshold: float = 1.0,
    split_high_weight: float = 1.0,
) -> nn.Module:
    """'combined'(저장소 공용 CombinedLoss), 'mae'(raw 스케일 L1), 'rmse_mape' 중 하나를 만든다.

    'mae'는 merged_model이 원래 쓰던 목적함수(``F.l1_loss``)와 동일하고, 'combined'는
    이 저장소의 다른 모든 모델과 손실을 맞추기 위한 것이다. 'mae'일 때 gamma/eps는
    쓰이지 않는다.

    'rmse_mape'는 lora 브랜치의 2-stage 학습에서 stage 2가 쓰던 손실로, 최적화 대상과
    보고 지표(RMSE/MAPE(+1))를 일치시킨다. 이것만 ``reduction='none'``이 불가능해
    스칼라를 돌려준다(:data:`SCALAR_LOSS_TYPES` 참고).

    'demand_split'은 그 선형 결합의 구조적 문제(모든 셀이 두 항에 동시에 기여해서 예측을
    쪼그라뜨리는 거래가 이득이 됨)를 고친 것으로, 셀을 실제 수요에 따라 한 항에만 넣는다
    — 낮은 수요는 MAPE, 높은 수요는 제곱오차. 요소별이라 스칼라 분기가 필요 없다.
    """

    if loss_type == 'combined':
        return CombinedLoss(gamma=gamma, eps=eps, reduction='none')
    if loss_type == 'mae':
        return nn.L1Loss(reduction='none')
    if loss_type == 'rmse_mape':
        return RmseMapeLoss(rmse_weight=rmse_weight)
    if loss_type == 'demand_split':
        return DemandSplitLoss(threshold=split_threshold, high_weight=split_high_weight)
    raise ValueError(
        f"알 수 없는 loss_type: {loss_type!r} "
        f"(가능: 'combined', 'mae', 'rmse_mape', 'demand_split')"
    )


__all__ = ['CombinedLoss', 'DemandSplitLoss', 'RmseMapeLoss', 'SCALAR_LOSS_TYPES', 'build_loss']
