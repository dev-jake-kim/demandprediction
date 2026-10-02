"""검색 encoder 학습 데이터: (예측 시점 t, 노드 n) 쌍, 라벨 bucket, 라벨 균형 샘플링."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def demand_bucket_bounds(value_cap: int) -> list[int]:
    """bucket 하한 ``0, 1, 2, 3, 4, 5, 7, 10, 14, 19, 25, …``(5 이후 간격 2, 3, 4, …)를 cap 미만까지,
    cap 이상은 마지막 bucket 하나."""

    if value_cap < 1:
        raise ValueError(f'value_cap은 1 이상이어야 함 (받음: {value_cap})')
    bounds = [0, 1, 2, 3, 4, 5]
    step = 2
    while bounds[-1] + step < value_cap:
        bounds.append(bounds[-1] + step)
        step += 1
    return [bound for bound in bounds if bound < value_cap] + [value_cap]


def select_value_cap(train_grid: np.ndarray, top_fraction: float = 0.005) -> int:
    """``value ≥ cap``의 비율이 ``top_fraction`` 이하가 되는 최소 정수 cap (train 구간 셀·시간 수요)."""

    values = np.asarray(train_grid).reshape(-1).astype(np.int64)
    at_least = np.cumsum(np.bincount(values)[::-1])[::-1] / values.size  # P(value ≥ c)
    passing = np.nonzero(at_least <= top_fraction)[0]
    return int(passing[0]) if passing.size else int(values.max()) + 1


def window_table(grid: Tensor, local_radius: int) -> Tensor:
    """``[T, H, W]`` → ``[T, N, P]`` 시점별 노드 창(격자 밖 0, 행 우선)."""

    window = 2 * local_radius + 1
    patches = F.unfold(F.pad(grid.unsqueeze(1), (local_radius,) * 4), kernel_size=window)
    return patches.transpose(1, 2).contiguous()


@dataclass
class PairSet:
    """한 split의 (t, n) 쌍(입력이 전부 0인 쌍 제외)."""

    times: Tensor   # [M] 예측 시점 t
    nodes: Tensor   # [M]
    labels: Tensor  # [M] 노드 자신의 t 시점 수요
    buckets: Tensor  # [M]


def build_pairs(
    crops: Tensor, grid_flat: Tensor, start: int, end: int, time_step: int, bounds: Tensor
) -> PairSet:
    """``t ∈ [max(start, k), end)``의 모든 노드 쌍 중 입력 창 ``[t−k, t)``가 0이 아닌 것."""

    first = max(start, time_step)
    times = torch.arange(first, end, device=crops.device)
    # 창 [t−k, t)의 합 > 0 인지: 시점별 창 합의 길이 k 이동합.
    per_step = crops.sum(dim=-1)  # [T, N]
    cumulative = torch.cat([per_step.new_zeros(1, per_step.shape[1]), per_step.cumsum(0)])
    window_sum = cumulative[times] - cumulative[times - time_step]  # [len, N]
    t_index, node_index = torch.nonzero(window_sum > 0, as_tuple=True)
    pair_times = times[t_index]
    labels = grid_flat[pair_times, node_index]
    buckets = torch.bucketize(labels, bounds, right=True) - 1
    return PairSet(pair_times, node_index, labels, buckets)


def gather_windows(crops: Tensor, times: Tensor, nodes: Tensor, time_step: int) -> Tensor:
    """``[M]`` (t, n) → ``[M, k, P]`` 창 ``[t−k, t)``."""

    steps = times.unsqueeze(1) + torch.arange(-time_step, 0, device=times.device)  # [M, k]
    return crops[steps, nodes.unsqueeze(1)]


def balanced_sample_weights(buckets: Tensor, num_buckets: int, power: float = 0.5) -> Tensor:
    """bucket g의 뽑힐 확률 ∝ ``count_g^power``가 되도록 샘플 가중치 ``count_g^(power−1)``."""

    counts = torch.bincount(buckets, minlength=num_buckets).float()
    per_bucket = torch.where(counts > 0, counts ** (power - 1.0), torch.zeros_like(counts))
    return per_bucket[buckets]


__all__ = [
    'PairSet',
    'balanced_sample_weights',
    'build_pairs',
    'demand_bucket_bounds',
    'gather_windows',
    'select_value_cap',
    'window_table',
]
