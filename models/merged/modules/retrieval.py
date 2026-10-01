"""Raw local-window retrieval with mode-dependent candidate masks, and its fusion into h_neural."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# 한 번에 처리하는 예측 시점 수. 메모리 상한(N·QUERY_CHUNK·retrieval_chunk_size)만 정한다.
QUERY_CHUNK = 64


class CausalRetrieval(nn.Module):
    """노드별 3×3(반경 r) raw 창으로 비슷한 과거 시점을 찾아 그 시점의 창 수요를 가중평균한다.

    - Q(t) = K(t) = 창 수요 ``[t-k, t)`` (``[k·P]``로 펴 L2 정규화), V(τ) = 창 수요 ``τ`` (``[P]``).
    - 후보 τ (모두 ``τ ≥ k``):
      - train 모드: ``τ < train_end`` 이고 ``τ ∉ [t, t + future_mask_hours]``
      - eval 모드: ``τ < t``
    - 결과는 ``(모드, t)``에만 의존하므로 모드별로 모든 t를 첫 사용 시 한 번 계산해 두고,
      forward는 ``sample_idx``로 gather만 한다(GPU→CPU 동기화 없음).
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
        # (training, device) -> [T, N, P]. 체크포인트에 저장하지 않는다.
        self._tables: dict[tuple[bool, torch.device], Tensor] = {}
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
        self._tables.clear()
        self.grid_path = str(path)

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
    def _build_table(self, training: bool, device: torch.device) -> Tensor:
        assert self._crops is not None
        crops = self._crops.to(device)
        total, nodes, neighbors = crops.shape
        k = self.time_step
        table = crops.new_zeros(total, nodes, neighbors)
        if total <= k:
            return table
        # windows[c] = 창 수요 [c, c+k) -> 시점 τ = c + k의 K이자 같은 t의 Q.
        windows = crops.unfold(0, k, 1)[: total - k]  # [C, N, P, k]
        count = windows.shape[0]
        keys = F.normalize(windows.reshape(count, nodes, neighbors * k), dim=-1)
        keys = keys.permute(1, 0, 2).contiguous()  # [N, C, D]
        values = crops[k:].permute(1, 0, 2).contiguous()  # [N, C, P] = V(c + k)
        times = torch.arange(k, total, device=device)
        node_index = torch.arange(nodes, device=device).view(-1, 1, 1)

        for q_start in range(0, count, QUERY_CHUNK):
            q_end = min(q_start + QUERY_CHUNK, count)
            queries = keys[:, q_start:q_end]  # [N, Q, D]
            best_scores: Tensor | None = None
            best_index: Tensor | None = None
            for c_start in range(0, count, self.retrieval_chunk_size):
                c_end = min(c_start + self.retrieval_chunk_size, count)
                scores = torch.bmm(queries, keys[:, c_start:c_end].transpose(1, 2))  # [N, Q, Cc]
                allowed = self.candidate_mask(times[q_start:q_end], times[c_start:c_end], training)
                scores = scores.masked_fill(~allowed, float('-inf'))
                take = min(self.retrieval_k, c_end - c_start)
                chunk_scores, chunk_index = scores.topk(take, dim=-1)
                chunk_index = chunk_index + c_start
                if best_scores is None:
                    best_scores, best_index = chunk_scores, chunk_index
                else:
                    merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
                    merged_index = torch.cat([best_index, chunk_index], dim=-1)
                    take = min(self.retrieval_k, merged_scores.shape[-1])
                    best_scores, order = merged_scores.topk(take, dim=-1)
                    best_index = merged_index.gather(-1, order)
            assert best_scores is not None and best_index is not None
            # 허용 후보가 하나도 없으면 softmax가 NaN -> 0 (검색 결과 0).
            weights = torch.softmax(best_scores, dim=-1).nan_to_num(0.0)  # [N, Q, K]
            picked = values[node_index, best_index]  # [N, Q, K, P]
            retrieved = (weights.unsqueeze(-1) * picked).sum(dim=2)  # [N, Q, P]
            table[k + q_start : k + q_end] = retrieved.transpose(0, 1)
        return table

    def table(self, training: bool, device: torch.device) -> Tensor | None:
        """``[T, N, P]`` 모드별 검색 결과 표. grid가 없으면 None."""

        if self._crops is None:
            return None
        key = (training, torch.device(device))
        cached = self._tables.get(key)
        if cached is None:
            cached = self._build_table(training, key[1])
            self._tables[key] = cached
        return cached

    def forward(self, sample_idx: Tensor) -> Tensor:
        """``[B]`` 절대 예측 시점 -> ``[B, N, P]`` 검색된 창 수요(raw). 모드는 ``self.training``."""

        table = self.table(self.training, sample_idx.device)
        if table is None:
            return torch.zeros(
                sample_idx.shape[0], self.num_nodes, self.num_neighbors, device=sample_idx.device
            )
        return table[sample_idx.long()]


class RetrievalFusion(nn.Module):
    """``h_local = Linear([h_neural ⊕ Linear(log1p(retrieved))])``."""

    def __init__(self, num_neighbors: int, history_hidden: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(num_neighbors, history_hidden)
        self.fuse = nn.Linear(2 * history_hidden, history_hidden)

    def forward(self, h_neural: Tensor, retrieved: Tensor) -> Tensor:
        r_emb = self.value_projection(torch.log1p(retrieved.to(h_neural.dtype)))
        return self.fuse(torch.cat([h_neural, r_emb], dim=-1))


__all__ = ['CausalRetrieval', 'RetrievalFusion']
