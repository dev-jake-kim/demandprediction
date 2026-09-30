"""Training objectives and loss metadata for the merged model.

Elementwise objectives use ``reduction='none'`` so the model can compute both element
mean and sum; ``rmse_mape`` returns a scalar.
"""

from __future__ import annotations

from torch import nn

from ..losses import CombinedLoss, DemandSplitLoss, RmseMapeLoss

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
) -> nn.Module:
    """Build a configured training objective by name.

    ``combined``, ``mae``, and ``demand_split`` return elementwise losses;
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
    raise ValueError(
        f"알 수 없는 loss_type: {loss_type!r} "
        f"(가능: 'combined', 'mae', 'rmse_mape', 'demand_split')"
    )


__all__ = ['CombinedLoss', 'DemandSplitLoss', 'RmseMapeLoss', 'SCALAR_LOSS_TYPES', 'build_loss']
