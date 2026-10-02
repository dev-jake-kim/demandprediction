"""Raw local-window retrieval with mode-dependent candidate masks, and its fusion into h_neural."""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# 한 번에 처리하는 예측 시점 수. 메모리 상한(N·QUERY_CHUNK·retrieval_chunk_size)만 정한다.
QUERY_CHUNK = 64


def demand_bucket_bounds(value_cap: int) -> list[int]:
    """수요 bucket의 하한 목록. 0~5는 1단위, 이후 간격이 2, 3, 4, ...로 커지고
    (0, 1, 2, 3, 4, 5, 7, 10, 14, 19, 25, ...), ``value_cap`` 이상은 마지막 bucket 하나다."""

    if value_cap < 1:
        raise ValueError(f'retrieval_value_cap은 1 이상이어야 함 (받음: {value_cap})')
    bounds = [0, 1, 2, 3, 4, 5]
    step = 2
    while bounds[-1] + step < value_cap:
        bounds.append(bounds[-1] + step)
        step += 1
    return [bound for bound in bounds if bound < value_cap] + [value_cap]


class CausalRetrieval(nn.Module):
    """노드별 3×3(반경 r) raw 창으로 비슷한 과거 시점 top-k를 찾아 그 유사도와 label을 돌려준다.

    - Q(t) = K(t) = 창 수요 ``[t-k, t)`` (``[k·P]``로 펴 L2 정규화), V(τ) = 노드 자신의 수요 ``τ``.
    - 후보 τ (모두 ``τ ≥ k``):
      - train 모드: ``τ < train_end`` 이고 ``τ ∉ [t, t + future_mask_hours]``
      - eval 모드: ``τ < t``
    - 결과는 ``(모드, t)``에만 의존하므로 모드별로 모든 t의 top-k(유사도, τ)를 첫 사용 시 한 번
      계산해 두고, forward는 ``sample_idx``로 gather만 한다(GPU→CPU 동기화 없음).
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        time_step: int,
        local_radius: int,
        retrieval_grid_path: str | Path | None,
        retrieval_k: int,
        retrieval_chunk_size: int,
        retrieval_train_end: int | None,
        future_mask_hours: int,
        query_measure: str = 'mean',
        spatial_sigma: float | None = None,
    ) -> None:
        super().__init__()
        if query_measure not in ('mean', 'nonzero_count'):
            raise ValueError(
                f"retrieval_query_measure는 'mean'|'nonzero_count' (받음: {query_measure!r})"
            )
        if spatial_sigma is not None and spatial_sigma <= 0:
            raise ValueError(f'retrieval_spatial_sigma는 양수 또는 null이어야 함 (받음: {spatial_sigma})')
        if future_mask_hours < time_step:
            raise ValueError(
                f'retrieval_future_mask_hours({future_mask_hours})는 time_step({time_step}) 이상이어야 '
                '함 - (t, t+k] 후보의 K 창에 정답 y_t가 들어간다'
            )
        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = time_step
        self.local_radius = local_radius
        self.window_size = 2 * local_radius + 1
        self.num_neighbors = self.window_size * self.window_size
        self.retrieval_k = retrieval_k
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieval_train_end = retrieval_train_end
        self.future_mask_hours = future_mask_hours
        self.query_measure = query_measure
        # 유사도의 칸별 가중치 w_p = exp(−d_p² / 2σ²) (d_p: 창 중심과의 거리, 행 우선). None이면 모두 1.
        # from_pretrained의 meta-device 초기화에서도 값이 남도록 tensor가 아닌 float로 둔다.
        offsets = range(-local_radius, local_radius + 1)
        self.spatial_weights = [
            1.0 if spatial_sigma is None else math.exp(-(dy * dy + dx * dx) / (2.0 * spatial_sigma ** 2))
            for dy in offsets for dx in offsets
        ]
        self._crops: Tensor | None = None
        # 체크포인트에 저장하지 않는 장치별 사본.
        self._device_crops: dict[torch.device, Tensor] = {}
        self._query_measures: dict[torch.device, Tensor] = {}
        # (training, device) -> (scores [T,N,K], times [T,N,K]).
        self._tables: dict[tuple[bool, torch.device], tuple[Tensor, Tensor]] = {}
        if retrieval_grid_path is not None:
            self._load_grid(retrieval_grid_path)

    def _load_grid(self, grid_path: str | Path) -> None:
        path = Path(grid_path).expanduser().resolve()
        grid = np.load(path, mmap_mode='r')
        if grid.ndim != 3 or tuple(grid.shape[1:]) != (self.height, self.width):
            raise ValueError(
                f'Retrieval grid must have shape [T,{self.height},{self.width}], got {grid.shape}'
            )
        if not np.isfinite(grid).all() or np.min(grid) < 0:
            raise ValueError('Retrieval grid must contain finite non-negative values')
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32, copy=True))
        padded = F.pad(grid_tensor.unsqueeze(1), (self.local_radius,) * 4)
        patches = F.unfold(padded, kernel_size=self.window_size)
        # [T, N, P]: 시점별 노드 창(격자 밖 0, 행 우선).
        self._crops = patches.transpose(1, 2).contiguous()
        self._device_crops.clear()
        self._query_measures.clear()
        self._tables.clear()
        self.grid_path = str(path)

    def crops(self, device: torch.device) -> Tensor:
        assert self._crops is not None
        device = torch.device(device)
        cached = self._device_crops.get(device)
        if cached is None:
            cached = self._crops.to(device)
            self._device_crops[device] = cached
        return cached

    def query_measures(self, device: torch.device) -> Tensor:
        """``[T, N]`` 질의 창(``[t-k, t)`` × 3×3 raw 수요, 격자 밖 0)의 m. ``t < k``는 0.

        ``query_measure='mean'``이면 ``k·P``개 값의 평균, ``'nonzero_count'``면 0이 아닌 칸 수.
        """

        device = torch.device(device)
        cached = self._query_measures.get(device)
        if cached is None:
            crops = self.crops(device)
            total, k = crops.shape[0], self.time_step
            cached = crops.new_zeros(total, crops.shape[1])
            if total > k:
                if self.query_measure == 'mean':
                    per_step = crops.mean(dim=-1)
                    reduce = 'mean'
                else:
                    per_step = (crops > 0).sum(dim=-1).to(crops.dtype)
                    reduce = 'sum'
                # unfold의 c번째 창 [c, c+k)는 시점 t = c + k의 질의다.
                windows = per_step.unfold(0, k, 1)[: total - k]
                cached[k:] = windows.mean(dim=-1) if reduce == 'mean' else windows.sum(dim=-1)
            self._query_measures[device] = cached
        return cached

    def candidate_mask(self, target_times: Tensor, candidate_times: Tensor, training: bool) -> Tensor:
        """``[Q]``, ``[C]`` -> ``[Q, C]`` bool. 예측 시점 t마다 후보 τ가 허용되는가."""

        t = target_times.view(-1, 1)
        tau = candidate_times.view(1, -1)
        if training:
            if self.retrieval_train_end is None:
                raise ValueError('train 모드 검색에는 retrieval_train_end가 필요함')
            return (tau < self.retrieval_train_end) & ((tau < t) | (tau > t + self.future_mask_hours))
        return tau < t

    @torch.no_grad()
    def _build_table(self, training: bool, device: torch.device) -> tuple[Tensor, Tensor]:
        crops = self.crops(device)
        total, nodes, neighbors = crops.shape
        k = self.time_step
        count = max(total - k, 0)
        top = max(min(self.retrieval_k, count), 1)
        # 허용 후보가 없는 칸은 score -inf (τ는 0, 쓰이지 않는다).
        scores = crops.new_full((total, nodes, top), float('-inf'))
        times = torch.zeros(total, nodes, top, dtype=torch.long, device=device)
        if count == 0:
            return scores, times
        # windows[c] = 창 수요 [c, c+k) -> 시점 τ = c + k의 K이자 같은 t의 Q.
        windows = crops.unfold(0, k, 1)[:count]  # [C, N, P, k]
        # 가중 cosine Σ w x y / (√Σ w x² · √Σ w y²): 칸마다 √w를 곱한 뒤 L2 정규화한다.
        scale = torch.tensor(self.spatial_weights, device=device).sqrt().view(1, 1, neighbors, 1)
        keys = F.normalize((windows * scale).reshape(count, nodes, neighbors * k), dim=-1)
        keys = keys.permute(1, 0, 2).contiguous()  # [N, C, D]
        candidate_times = torch.arange(k, total, device=device)

        for q_start in range(0, count, QUERY_CHUNK):
            q_end = min(q_start + QUERY_CHUNK, count)
            queries = keys[:, q_start:q_end]  # [N, Q, D]
            best_scores: Tensor | None = None
            best_index: Tensor | None = None
            for c_start in range(0, count, self.retrieval_chunk_size):
                c_end = min(c_start + self.retrieval_chunk_size, count)
                chunk = torch.bmm(queries, keys[:, c_start:c_end].transpose(1, 2))  # [N, Q, Cc]
                allowed = self.candidate_mask(
                    candidate_times[q_start:q_end], candidate_times[c_start:c_end], training
                )
                chunk = chunk.masked_fill(~allowed, float('-inf'))
                chunk_scores, chunk_index = chunk.topk(min(top, c_end - c_start), dim=-1)
                chunk_index = chunk_index + c_start
                if best_scores is None:
                    best_scores, best_index = chunk_scores, chunk_index
                else:
                    merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
                    merged_index = torch.cat([best_index, chunk_index], dim=-1)
                    best_scores, order = merged_scores.topk(min(top, merged_scores.shape[-1]), dim=-1)
                    best_index = merged_index.gather(-1, order)
            assert best_scores is not None and best_index is not None
            width = best_scores.shape[-1]
            scores[k + q_start : k + q_end, :, :width] = best_scores.transpose(0, 1)
            picked = torch.where(torch.isfinite(best_scores), best_index + k, 0)
            times[k + q_start : k + q_end, :, :width] = picked.transpose(0, 1)
        return scores, times

    def table(self, training: bool, device: torch.device) -> tuple[Tensor, Tensor] | None:
        """모드별 top-k 표 ``(scores [T,N,K], times [T,N,K])``. grid가 없으면 None."""

        if self._crops is None:
            return None
        key = (training, torch.device(device))
        cached = self._tables.get(key)
        if cached is None:
            cached = self._build_table(training, key[1])
            self._tables[key] = cached
        return cached

    def forward(self, sample_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``[B]`` 절대 예측 시점 -> ``(scores [B,N,K], values [B,N,K], query_measure [B,N])``.
        모드는 ``self.training``. ``values``는 후보 τ에서 예측 대상 노드 자신의 수요(label)다.

        허용 후보가 없는 칸은 score가 ``-inf``다(values는 의미 없음).
        """

        device = sample_idx.device
        table = self.table(self.training, device)
        batch = sample_idx.shape[0]
        if table is None:
            return (
                torch.full((batch, self.num_nodes, 1), float('-inf'), device=device),
                torch.zeros(batch, self.num_nodes, 1, device=device),
                torch.zeros(batch, self.num_nodes, device=device),
            )
        scores, times = table
        index = sample_idx.long()
        picked_times = times[index]  # [B, N, K]
        node_index = torch.arange(self.num_nodes, device=device).view(1, -1, 1)
        # 창은 행 우선이므로 가운데 칸(P // 2)이 노드 자신이다.
        values = self.crops(device)[picked_times, node_index, self.num_neighbors // 2]  # [B, N, K]
        return scores[index], values, self.query_measures(device)[index]


FUSION_MODES = ('average', 'embedding', 'vote', 'average_gate')


def weighted_label_mean(scores: Tensor, values: Tensor) -> Tensor:
    """``softmax(s)``로 유효 후보 label을 가중평균한다 ``[B,N,K] -> [B,N,1]``. 후보가 없으면 0."""

    valid = torch.isfinite(scores)
    # 허용 후보가 하나도 없으면 softmax가 NaN -> 0 (검색 결과 0).
    weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return (weights * values.masked_fill(~valid, 0.0)).sum(dim=-1, keepdim=True)


class RetrievalGate(nn.Module):
    """``retrieval_fusion='average_gate'``: BranchAttention 뒤의 노드 표현 z에 검색 결과를 섞는다.

    ``r = Linear(1 → fusion_dim)(log1p(softmax(s) 가중평균 label))``,
    ``g = sigmoid(Linear(fusion_dim → 1)(z))``, 출력 ``g·r + (1 − g)·z``.
    """

    def __init__(self, fusion_dim: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, fusion_dim)
        self.gate = nn.Linear(fusion_dim, 1)

    def forward(self, z: Tensor, scores: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """``(z [B,N,F], scores, values) -> (blended [B,N,F], gate [B,N])``."""

        retrieved = self.value_projection(torch.log1p(weighted_label_mean(scores, values).to(z.dtype)))
        gate = torch.sigmoid(self.gate(z))
        return gate * retrieved + (1.0 - gate) * z, gate.squeeze(-1)


class RetrievalFusion(nn.Module):
    """검색 후보 (유사도 s, label v)를 노드 벡터로 만들어 ``h_neural``과 concat하고 ``Linear``로
    ``history_hidden``에 맞춘다.

    - ``average``: ``softmax(s)``로 v를 가중평균하고 ``Linear(1 → H)(log1p(·))``로 편다.
    - ``embedding``: 후보 합 ``C = Σ sigmoid(a·s + b) · Embedding(bucket(v))`` (E차원).
      ``use_fallback``이면 질의 크기 m으로 학습 벡터 O와 ``(1 − w)·C + w·O`` (``w = e^{−m/τ}``)로
      섞는다. 질의가 전부 0이면(m=0) 유사도가 모두 0이라 top-k가 무의미하므로 결과는 O가 된다.
      ``learn_tau``면 τ를 ``log τ`` 파라미터로 학습한다.
    - ``vote``: v를 가장 가까운 bucket 값으로 반올림하고(동점은 위쪽, cap 이상은 마지막 bucket)
      bucket마다 그 후보들의 s를 합한다. 수요 0 bucket을 뺀 ``[B, N, buckets − 1]``을 쓴다.
    """

    def __init__(
        self,
        history_hidden: int,
        *,
        mode: str = 'embedding',
        embedding_dim: int = 16,
        value_cap: int | None = None,
        use_fallback: bool = True,
        fallback_tau: float = 1.0,
        learn_tau: bool = False,
    ) -> None:
        super().__init__()
        if mode not in ('average', 'embedding', 'vote'):
            raise ValueError(
                f"RetrievalFusion mode는 'average'|'embedding'|'vote' (average_gate는 RetrievalGate, 받음: {mode!r})"
            )
        self.mode = mode
        self.use_fallback = use_fallback and mode == 'embedding'
        if mode == 'average':
            self.value_projection = nn.Linear(1, history_hidden)
            self.fuse = nn.Linear(2 * history_hidden, history_hidden)
            return
        if value_cap is None:
            raise ValueError(f'{mode} 융합에는 retrieval_value_cap이 필요함')
        bounds = demand_bucket_bounds(value_cap)
        # persistent: from_pretrained의 meta-device 초기화는 비저장 buffer를 복원하지 않는다.
        self.register_buffer('bucket_bounds', torch.tensor(bounds, dtype=torch.float32))
        if mode == 'vote':
            self.fuse = nn.Linear(history_hidden + len(bounds) - 1, history_hidden)
            return
        if fallback_tau <= 0:
            raise ValueError(f'retrieval_fallback_tau는 양수여야 함 (받음: {fallback_tau})')
        self.value_embedding = nn.Embedding(len(bounds), embedding_dim)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))
        if self.use_fallback:
            # O: 질의 수요가 적을수록 검색 결과 대신 쓰는 학습 벡터. 0에서 시작한다.
            self.fallback_embedding = nn.Parameter(torch.zeros(embedding_dim))
            self.fallback_log_tau = (
                nn.Parameter(torch.tensor(math.log(fallback_tau))) if learn_tau else None
            )
            self.fallback_tau = float(fallback_tau)
        self.fuse = nn.Linear(history_hidden + embedding_dim, history_hidden)

    def fallback_weight(self, query_measure: Tensor) -> Tensor:
        """``w = e^{−m/τ}``."""

        if self.fallback_log_tau is None:
            return torch.exp(-query_measure / self.fallback_tau)
        return torch.exp(-query_measure * torch.exp(-self.fallback_log_tau))

    def bucketize(self, values: Tensor) -> Tensor:
        """embedding용: 하한 기준 bucket(내림)."""

        return torch.bucketize(values, self.bucket_bounds, right=True) - 1

    def nearest_bucket(self, values: Tensor) -> Tensor:
        """vote용: 가장 가까운 bucket 값의 index. 동점이면 위쪽, cap 이상은 마지막 bucket."""

        bounds = self.bucket_bounds
        lower = self.bucketize(values)
        upper = (lower + 1).clamp_max(len(bounds) - 1)
        take_upper = (bounds[upper] - values) <= (values - bounds[lower])
        return torch.where(take_upper, upper, lower)

    def forward(self, h_neural: Tensor, scores: Tensor, values: Tensor, query_measure: Tensor) -> Tensor:
        valid = torch.isfinite(scores)
        if self.mode == 'average':
            averaged = weighted_label_mean(scores, values)
            r_emb = self.value_projection(torch.log1p(averaged.to(h_neural.dtype)))
            return self.fuse(torch.cat([h_neural, r_emb], dim=-1))
        if self.mode == 'vote':
            weight = scores.masked_fill(~valid, 0.0).to(h_neural.dtype)  # [B, N, K]
            votes = weight.new_zeros(*weight.shape[:-1], len(self.bucket_bounds))
            votes = votes.scatter_add(-1, self.nearest_bucket(values), weight)
            # 수요 0 bucket(index 0)은 버린다.
            return self.fuse(torch.cat([h_neural, votes[..., 1:]], dim=-1))
        gate = torch.sigmoid(self.score_scale * scores.masked_fill(~valid, 0.0) + self.score_bias)
        gate = (gate * valid).to(h_neural.dtype)  # [B, N, K]
        embedded = self.value_embedding(self.bucketize(values))  # [B, N, K, E]
        retrieved = (gate.unsqueeze(-1) * embedded).sum(dim=2)  # C [B, N, E]
        if self.use_fallback:
            weight = self.fallback_weight(query_measure.to(h_neural.dtype)).unsqueeze(-1)
            retrieved = (1.0 - weight) * retrieved + weight * self.fallback_embedding
        return self.fuse(torch.cat([h_neural, retrieved], dim=-1))


__all__ = [
    'FUSION_MODES',
    'CausalRetrieval',
    'RetrievalFusion',
    'RetrievalGate',
    'demand_bucket_bounds',
    'weighted_label_mean',
]
