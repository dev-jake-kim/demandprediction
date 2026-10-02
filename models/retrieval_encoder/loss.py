"""라벨 거리 가중 contrastive(latent 유클리드 거리) + 표준 정규 KL (docs/RETRIEVAL_ENCODER.md §4)."""

from __future__ import annotations

import torch
from torch import Tensor


def weighted_contrastive_loss(
    z: Tensor, labels: Tensor, buckets: Tensor, temperature: float
) -> tuple[Tensor, int]:
    """``L_con``과 anchor 수.

    ``s_ij = −‖z_i − z_j‖² / (L·T)``. 같은 bucket(자기 자신 제외)이 positive, 다른 bucket은
    ``w_ij = |log1p(y_i) − log1p(y_j)|`` 가중 negative. positive가 없는 샘플은 anchor에서 뺀다.
    """

    batch, latent = z.shape
    # cdist는 거리 0(대각)에서 backward가 NaN이 될 수 있어 제곱거리를 직접 계산한다.
    squared = z.pow(2).sum(dim=1)
    distance_sq = (squared.unsqueeze(0) + squared.unsqueeze(1) - 2.0 * z @ z.T).clamp_min(0.0)
    scores = -distance_sq / (latent * temperature)
    eye = torch.eye(batch, dtype=torch.bool, device=z.device)
    same = (buckets.unsqueeze(0) == buckets.unsqueeze(1)) & ~eye
    log_label = torch.log1p(labels.float())
    weights = (log_label.unsqueeze(0) - log_label.unsqueeze(1)).abs()
    # 분모 가중치: positive 1, negative w_ij, 자기 자신 0 → log 공간에서 더한다(log 0 = −inf).
    denom_weight = torch.where(same, torch.ones_like(weights), weights).masked_fill(eye, 0.0)
    log_denominator = torch.logsumexp(scores + denom_weight.log(), dim=1)
    positives = same.float()
    count = positives.sum(dim=1)
    anchors = count > 0
    if not bool(anchors.any()):
        return z.sum() * 0.0, 0
    log_prob = scores - log_denominator.unsqueeze(1)
    per_anchor = -(log_prob * positives).sum(dim=1)[anchors] / count[anchors]
    return per_anchor.mean(), int(anchors.sum())


def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """``mean_i KL(N(μ_i, σ_i²) ‖ N(0, I))``."""

    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=1).mean()


__all__ = ['gaussian_kl', 'weighted_contrastive_loss']
