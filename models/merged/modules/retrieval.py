"""Raw local-window retrieval with mode-dependent candidate masks, and its fusion into h_neural."""

from __future__ import annotations

from pathlib import Path

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
    """노드별 3×3(반경 r) raw 창으로 비슷한 과거 시점 top-k를 찾아 그 유사도와 창 수요를 돌려준다.

    - Q(t) = K(t) = 창 수요 ``[t-k, t)`` (``[k·P]``로 펴 L2 정규화), V(τ) = 창 수요 ``τ`` (``[P]``).
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
    ) -> None:
        super().__init__()
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
        self._crops: Tensor | None = None
        # 체크포인트에 저장하지 않는 장치별 사본.
        self._device_crops: dict[torch.device, Tensor] = {}
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
        keys = F.normalize(windows.reshape(count, nodes, neighbors * k), dim=-1)
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

    def forward(self, sample_idx: Tensor) -> tuple[Tensor, Tensor]:
        """``[B]`` 절대 예측 시점 -> ``(scores [B,N,K], values [B,N,K,P])``. 모드는 ``self.training``.

        허용 후보가 없는 칸은 score가 ``-inf``다(values는 의미 없음).
        """

        device = sample_idx.device
        table = self.table(self.training, device)
        if table is None:
            batch = sample_idx.shape[0]
            return (
                torch.full((batch, self.num_nodes, 1), float('-inf'), device=device),
                torch.zeros(batch, self.num_nodes, 1, self.num_neighbors, device=device),
            )
        scores, times = table
        index = sample_idx.long()
        picked_times = times[index]  # [B, N, K]
        node_index = torch.arange(self.num_nodes, device=device).view(1, -1, 1)
        values = self.crops(device)[picked_times, node_index]  # [B, N, K, P]
        return scores[index], values


class RetrievalFusion(nn.Module):
    """검색 후보마다 ``sigmoid(a·s + b) · Embedding(bucket(v))``를 만들어 후보 합을 구하고
    ``h_neural``과 concat해 ``Linear``로 ``history_hidden``에 맞춘다.

    ``v``는 후보 τ의 3×3 창 수요 ``P``개이며 칸별로 embedding을 따로 합한 뒤 ``[P·E]``로 편다.
    """

    def __init__(
        self, num_neighbors: int, history_hidden: int, embedding_dim: int, value_cap: int
    ) -> None:
        super().__init__()
        bounds = demand_bucket_bounds(value_cap)
        # persistent: from_pretrained의 meta-device 초기화는 비저장 buffer를 복원하지 않는다.
        self.register_buffer('bucket_bounds', torch.tensor(bounds, dtype=torch.float32))
        self.value_embedding = nn.Embedding(len(bounds), embedding_dim)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))
        self.fuse = nn.Linear(history_hidden + num_neighbors * embedding_dim, history_hidden)

    def bucketize(self, values: Tensor) -> Tensor:
        return torch.bucketize(values, self.bucket_bounds, right=True) - 1

    def forward(self, h_neural: Tensor, scores: Tensor, values: Tensor) -> Tensor:
        valid = torch.isfinite(scores)
        gate = torch.sigmoid(self.score_scale * scores.masked_fill(~valid, 0.0) + self.score_bias)
        gate = (gate * valid).to(h_neural.dtype)  # [B, N, K]
        embedded = self.value_embedding(self.bucketize(values))  # [B, N, K, P, E]
        summed = (gate[..., None, None] * embedded).sum(dim=2)  # [B, N, P, E]
        return self.fuse(torch.cat([h_neural, summed.flatten(2)], dim=-1))


__all__ = ['CausalRetrieval', 'RetrievalFusion', 'demand_bucket_bounds']
