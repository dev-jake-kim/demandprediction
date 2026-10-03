"""Training objectives and loss metadata for the merged model.

Elementwise objectives use ``reduction='none'`` so the model can compute both element
mean and sum; ``rmse_mape`` returns a scalar.
"""

from __future__ import annotations

from torch import nn

from ..losses import CombinedLoss, DemandSplitLoss, MseMapeLoss, NodeMixLoss, RmseMapeLoss

# Scalar objectives bypass elementwise aggregation in the model.
SCALAR_LOSS_TYPES = frozenset({'rmse_mape'})


def build_loss(
    loss_type: str,
    *,
    gamma: float = 1.0,
    eps: float = 0.5,
    rmse_weight: float = 10.0,
    split_threshold: float = 1.0,
    split_high_weight: float = 1.0,
    mape_weight: float = 1.0,
    node_mix: tuple[list[float], list[float], list[float]] | None = None,
) -> nn.Module:
    """Build a configured training objective by name.

    ``combined``, ``mae``, ``demand_split``, ``mse_mape`` and ``node_mix`` return elementwise losses;
    ``rmse_mape`` returns a scalar.
    """

    if loss_type == 'combined':
        return CombinedLoss(gamma=gamma, eps=eps, reduction='none')
    if loss_type == 'mae':
        return nn.L1Loss(reduction='none')
    if loss_type == 'rmse_mape':
        return RmseMapeLoss(rmse_weight=rmse_weight)
    if loss_type == 'demand_split':
        return DemandSplitLoss(threshold=split_threshold, high_weight=split_high_weight)
    if loss_type == 'mse_mape':
        return MseMapeLoss(mape_weight=mape_weight)
    if loss_type == 'node_mix':
        if node_mix is None:
            raise ValueError('node_mix 손실에는 노드별 (lambda, mse_scale, weight)가 필요함 — train.py가 계산해 config에 넣는다')
        return NodeMixLoss(*node_mix)
    raise ValueError(
        f"알 수 없는 loss_type: {loss_type!r} "
        f"(가능: 'combined', 'mae', 'rmse_mape', 'demand_split', 'mse_mape', 'node_mix')"
    )


__all__ = ['CombinedLoss', 'DemandSplitLoss', 'MseMapeLoss', 'NodeMixLoss', 'RmseMapeLoss', 'SCALAR_LOSS_TYPES', 'build_loss']
