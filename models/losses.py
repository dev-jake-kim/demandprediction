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
    """Scalar ``rmse_weight * RMSE + MAPE(+1)`` objective.

    MAPE uses denominator ``abs(y_true) + 1`` and percent units. The RMSE spans all
    input elements, so this loss always returns a scalar.
    """

    def __init__(self, rmse_weight: float = 10.0) -> None:
        super().__init__()
        if not math.isfinite(rmse_weight) or rmse_weight < 0:
            raise ValueError(f'rmse_weight는 0 이상이어야 함: got {rmse_weight}')
        self.rmse_weight = rmse_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        # vector_norm avoids undefined 0/0 gradients at zero error.
        rmse = torch.linalg.vector_norm(diff) / math.sqrt(diff.numel())
        mape = (diff.abs() / (y_true.abs() + 1.0)).mean() * 100.0
        return self.rmse_weight * rmse + mape


class DemandSplitLoss(nn.Module):
    """Elementwise loss that uses MAPE(+1) for low-demand cells and weighted squared error
    above the threshold.

    ``y_true <= threshold`` uses ``abs(error) / (abs(y_true) + 1)``; higher values use
    ``high_weight * error**2``.
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


class MseMapeLoss(nn.Module):
    """Elementwise ``error**2 + mape_weight * abs(error) / (abs(y_true) + 1)``.

    The squared term pulls predictions toward the conditional mean (RMSE); the relative term
    pulls low-demand cells toward zero (MAPE(+1)). ``mape_weight`` trades the two off.
    """

    def __init__(self, mape_weight: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(mape_weight) or mape_weight < 0:
            raise ValueError(f'mape_weight는 0 이상이어야 함: got {mape_weight}')
        self.mape_weight = mape_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_true - y_pred
        return diff ** 2 + self.mape_weight * diff.abs() / (y_true.abs() + 1.0)


class NodeMixLoss(nn.Module):
    """Node-wise ``weight_n * (lam_n * |d| / (|y| + 1) * mape_scale_n + (1 - lam_n) * d**2 * mse_scale_n)``.

    ``mape_scale_n = 1 / B_n`` and ``mse_scale_n = 1 / (2 A_n)`` match both terms' gradient magnitudes
    at reference residuals, so ``lam_n`` only changes the loss shape within a node; ``weight_n``
    fixes the node's expected gradient magnitude (docs/SPEC.md §7). Per-node vectors have length
    ``H*W`` and are reshaped to the prediction grid.
    """

    def __init__(
        self, lam: list[float], mape_scale: list[float], mse_scale: list[float], weight: list[float]
    ) -> None:
        super().__init__()
        if not (len(lam) == len(mape_scale) == len(mse_scale) == len(weight)):
            raise ValueError('node_mix 벡터 길이가 다름')
        for name, values in (('lam', lam), ('mape_scale', mape_scale), ('mse_scale', mse_scale), ('weight', weight)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32), persistent=False)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        grid = y_pred.shape[-2:]
        if self.lam.device != y_pred.device:
            self.to(y_pred.device)
        lam, mape_scale, mse_scale, weight = (
            v.view(grid) for v in (self.lam, self.mape_scale, self.mse_scale, self.weight)
        )
        diff = y_true - y_pred
        relative = diff.abs() / (y_true.abs() + 1.0)
        return weight * (lam * relative * mape_scale + (1.0 - lam) * diff ** 2 * mse_scale)
