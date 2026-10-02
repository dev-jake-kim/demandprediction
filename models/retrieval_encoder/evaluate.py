"""검색 encoder 단독 평가 (docs/RETRIEVAL_ENCODER.md §7).

질의: split 구간의 (t, n) 중 입력 창이 0이 아닌 것. 후보: 같은 노드의 ``τ ∈ [k, t)``(입력 0 포함).
top-k를 골라 ``softmax(유사도)`` 가중평균 라벨로 예측하고 MAE·RMSE·같은 bucket 비율을 낸다.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .data import PairSet, gather_windows
from .encoder import RetrievalEncoder


@torch.no_grad()
def encode_all(encoder: RetrievalEncoder, crops: Tensor, time_step: int, batch: int = 65536) -> Tensor:
    """모든 ``t ≥ k``, 모든 노드의 μ ``[T, N, L]`` (``t < k``는 0). eval 모드로 계산한다(잡음 없음)."""

    was_training = encoder.training
    encoder.eval()
    total, nodes, _ = crops.shape
    out = crops.new_zeros(total, nodes, encoder.latent_dim)
    t_grid, n_grid = torch.meshgrid(
        torch.arange(time_step, total, device=crops.device), torch.arange(nodes, device=crops.device),
        indexing='ij',
    )
    times, node_ids = t_grid.reshape(-1), n_grid.reshape(-1)
    for start in range(0, len(times), batch):
        t, n = times[start:start + batch], node_ids[start:start + batch]
        mu, _ = encoder(gather_windows(crops, t, n, time_step), n)
        out[t, n] = mu
    encoder.train(was_training)
    return out


def raw_keys(crops: Tensor, time_step: int) -> Tensor:
    """기존 검색기의 질의·후보 벡터: 창 ``[t−k, t)``를 편 뒤 L2 정규화 ``[T, N, k·P]``."""

    total, nodes, neighbors = crops.shape
    out = crops.new_zeros(total, nodes, time_step * neighbors)
    windows = crops.unfold(0, time_step, 1)[: total - time_step]  # [C, N, P, k], 창 [c, c+k)
    flat = windows.reshape(total - time_step, nodes, -1)
    out[time_step:] = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return out


@torch.no_grad()
def retrieval_metrics(
    keys: Tensor,
    similarity,
    queries: PairSet,
    grid_flat: Tensor,
    bounds: Tensor,
    time_step: int,
    top_k: int = 20,
    node_chunk: int = 8,
) -> dict[str, float]:
    """``keys [T, N, E]``와 ``similarity(q [..., E], k [..., E]) -> [...]``로 질의를 검색해 평가한다."""

    total, nodes, _ = keys.shape
    candidate_times = torch.arange(total, device=keys.device)
    predictions = torch.zeros_like(queries.labels, dtype=torch.float32)
    same_bucket = torch.zeros_like(predictions)
    label_buckets = torch.bucketize(grid_flat, bounds, right=True) - 1  # [T, N]
    for node_start in range(0, nodes, node_chunk):
        node_end = min(node_start + node_chunk, nodes)
        for node in range(node_start, node_end):
            mask = queries.nodes == node
            if not bool(mask.any()):
                continue
            q_times = queries.times[mask]
            scores = similarity(keys[q_times, node].unsqueeze(1), keys[:, node].unsqueeze(0))  # [Q, T]
            allowed = (candidate_times.unsqueeze(0) >= time_step) & (candidate_times.unsqueeze(0) < q_times.unsqueeze(1))
            scores = scores.masked_fill(~allowed, float('-inf'))
            top_scores, top_times = scores.topk(min(top_k, total), dim=-1)
            valid = torch.isfinite(top_scores)
            weights = torch.softmax(top_scores, dim=-1).nan_to_num(0.0)
            values = grid_flat[top_times, node]
            predictions[mask] = (weights * values.masked_fill(~valid, 0.0)).sum(dim=-1)
            hits = (label_buckets[top_times, node] == queries.buckets[mask].unsqueeze(1)) & valid
            same_bucket[mask] = hits.float().sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    error = predictions - queries.labels
    result = {
        'mae': float(error.abs().mean()),
        'rmse': float(error.pow(2).mean().sqrt()),
        'same_bucket_ratio': float(same_bucket.mean()),
        'queries': int(len(error)),
    }
    for bucket in range(len(bounds)):
        mask = queries.buckets == bucket
        if bool(mask.any()):
            result[f'same_bucket_ratio_b{int(bounds[bucket])}'] = float(same_bucket[mask].mean())
            result[f'mae_b{int(bounds[bucket])}'] = float(error[mask].abs().mean())
    return result


def cosine_similarity(query: Tensor, keys: Tensor) -> Tensor:
    """정규화된 벡터의 내적 = cosine (기존 검색기와 같은 유사도·softmax 척도)."""

    return (query * keys).sum(dim=-1)


__all__ = ['cosine_similarity', 'encode_all', 'raw_keys', 'retrieval_metrics']
