"""Training objectives for the merged model.

원본 ``merged_model/losses.py``는 저장소 공용 ``CombinedLoss``를 파일 안에 복사해 두고
있었다. 여기서는 복사본 대신 ``models/losses.py``의 원본을 그대로 재사용한다 —
``reduction='none'``이면 두 구현의 반환값이 완전히 동일하다(요소별 텐서).

모든 손실은 ``reduction='none'``이다. 모델이 한 번만 reduce하므로 epoch 집계가
배치 평균의 평균이 아니라 진짜 요소 평균이 된다(``drop_last=False``라 배치 크기가 다르다).
"""

from __future__ import annotations

from torch import nn

from ..losses import CombinedLoss


def build_loss(loss_type: str, *, gamma: float = 1.0, eps: float = 0.5) -> nn.Module:
    """'combined'(저장소 공용 CombinedLoss) 또는 'mae'(raw 스케일 L1)를 만든다.

    'mae'는 merged_model이 원래 쓰던 목적함수(``F.l1_loss``)와 동일하고, 'combined'는
    이 저장소의 다른 모든 모델과 손실을 맞추기 위한 것이다. 'mae'일 때 gamma/eps는
    쓰이지 않는다.
    """

    if loss_type == 'combined':
        return CombinedLoss(gamma=gamma, eps=eps, reduction='none')
    if loss_type == 'mae':
        return nn.L1Loss(reduction='none')
    raise ValueError(f"알 수 없는 loss_type: {loss_type!r} (가능: 'combined', 'mae')")


__all__ = ['CombinedLoss', 'build_loss']
