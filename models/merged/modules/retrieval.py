"""Raw local-window retrieval with a causal time boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalRetrieval(nn.Module):
    """Retrieve raw demand values only from candidates ``tau < target_time``."""

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
        retrieval_scope: Literal["observed_past", "train_prefix"],
        retrieval_train_end: int | None,
    ) -> None:
        super().__init__()
        if retrieval_scope not in {"observed_past", "train_prefix"}:
            raise ValueError("retrieval_scope must be 'observed_past' or 'train_prefix'")
        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = time_step
        self.local_radius = local_radius
        self.window_size = 2 * local_radius + 1
        self.num_neighbors = self.window_size * self.window_size
        self.flat_query_dim = time_step * self.num_neighbors
        self.retrieval_k = retrieval_k
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieval_scope = retrieval_scope
        self.retrieval_train_end = retrieval_train_end
        self._grid: Tensor | None = None
        self._crops: Tensor | None = None
        self._cache: Tensor | None = None
        if retrieval_grid_path is not None:
            self._load_grid(retrieval_grid_path)

    def _load_grid(self, grid_path: str | Path) -> None:
        path = Path(grid_path).expanduser().resolve()
        grid = np.load(path, mmap_mode="r")
        if grid.ndim != 3 or tuple(grid.shape[1:]) != (self.height, self.width):
            raise ValueError(
                f"Retrieval grid must have shape [T,{self.height},{self.width}], got {grid.shape}"
            )
        if not np.isfinite(grid).all() or np.min(grid) < 0:
            raise ValueError("Retrieval grid must contain finite non-negative values")
        grid_tensor = torch.from_numpy(np.array(grid, dtype=np.float32, copy=True))
        self._grid = grid_tensor
        padded = F.pad(grid_tensor.unsqueeze(1), (self.local_radius,) * 4)
        patches = F.unfold(padded, kernel_size=self.window_size)
        self._crops = patches.transpose(1, 2).contiguous()
        # Keep the cache on CPU during meta-device initialization.
        self._cache = grid_tensor.new_full((grid_tensor.shape[0], self.num_nodes), float("nan"))
        self.grid_path = str(path)

    def _crop(self, demands: Tensor) -> Tensor:
        """``[B, k, H, W]`` -> ``[B, k, N, (2r+1)^2]`` 노드별 raw 창(격자 밖 0, 행 우선)."""

        batch, steps = demands.shape[:2]
        padded = F.pad(
            demands.reshape(batch * steps, 1, self.height, self.width), (self.local_radius,) * 4
        )
        patches = F.unfold(padded, kernel_size=self.window_size)
        return patches.transpose(1, 2).reshape(batch, steps, self.num_nodes, self.num_neighbors)

    @torch.no_grad()
    def forward(self, demands: Tensor, sample_idx: Tensor) -> Tensor:
        """``[B, k, H, W]`` 최근 수요로 인과 후보를 검색해 ``[B, N]`` raw 값을 반환한다."""

        if demands.ndim != 4 or tuple(demands.shape[1:]) != (self.time_step, self.height, self.width):
            raise ValueError("Unexpected retrieval query shape")
        local_crop = self._crop(demands)
        batch, steps, nodes, neighbors = local_crop.shape
        if self._grid is None or self._crops is None:
            return local_crop.new_zeros((batch, nodes))

        search_device = local_crop.device
        query = local_crop.detach().to(device=search_device, dtype=torch.float32)
        query = query.permute(0, 2, 3, 1).reshape(batch, nodes, self.flat_query_dim)
        query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sample_times = sample_idx.detach().to(device="cpu", dtype=torch.long).tolist()
        total_steps = self._grid.shape[0]
        result = torch.zeros(batch, nodes, dtype=torch.float32)
        uncached_positions: list[int] = []
        uncached_times: list[int] = []

        for position, target_time in enumerate(sample_times):
            target_time = int(target_time)
            if self._cache is not None and 0 <= target_time < self._cache.shape[0]:
                cached = self._cache[target_time]
                if torch.isfinite(cached).all():
                    result[position] = cached
                    continue
            uncached_positions.append(position)
            uncached_times.append(target_time)

        if not uncached_positions:
            return result.to(device=local_crop.device)

        start = self.time_step
        end_limits = []
        for target_time in uncached_times:
            end = target_time
            if self.retrieval_scope == "train_prefix":
                if self.retrieval_train_end is None:
                    raise ValueError("retrieval_train_end is required for train_prefix retrieval")
                end = min(end, int(self.retrieval_train_end))
            end_limits.append(min(end, total_steps))
        max_end = max(end_limits)
        if max_end <= start:
            for target_time in uncached_times:
                if self._cache is not None and 0 <= target_time < total_steps:
                    self._cache[target_time] = 0.0
            return result.to(device=local_crop.device)

        query_batch = query[uncached_positions]
        end_tensor = torch.tensor(end_limits, dtype=torch.long, device=search_device)
        best_scores: Tensor | None = None
        best_times: Tensor | None = None
        for chunk_start in range(start, max_end, self.retrieval_chunk_size):
            chunk_end = min(chunk_start + self.retrieval_chunk_size, max_end)
            candidate_times = torch.arange(chunk_start, chunk_end, dtype=torch.long, device=search_device)
            base = self._crops[chunk_start - self.time_step : chunk_end - 1]
            windows = base.unfold(0, self.time_step, 1)
            candidates = windows.permute(1, 0, 2, 3).reshape(nodes, -1, self.flat_query_dim)
            candidates = candidates.to(device=search_device)
            candidate_norm = candidates.norm(dim=-1).clamp_min(1e-8)
            similarity = torch.einsum("bnd,ncd->bnc", query_batch, candidates) / candidate_norm
            similarity = similarity.nan_to_num(0.0, 0.0, 0.0)
            valid_candidates = candidate_times.view(1, 1, -1) < end_tensor.view(-1, 1, 1)
            similarity = similarity.masked_fill(~valid_candidates, torch.finfo(similarity.dtype).min)

            take = min(self.retrieval_k, similarity.shape[-1])
            chunk_scores, chunk_indices = similarity.topk(take, dim=-1)
            chunk_times = candidate_times[chunk_indices]
            if best_scores is None:
                best_scores, best_times = chunk_scores, chunk_times
            else:
                merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
                merged_times = torch.cat([best_times, chunk_times], dim=-1)
                take = min(self.retrieval_k, merged_scores.shape[-1])
                best_scores, order = merged_scores.topk(take, dim=-1)
                best_times = merged_times.gather(-1, order)

        assert best_scores is not None and best_times is not None
        has_candidates = end_tensor > start
        valid_best = best_times < end_tensor.view(-1, 1, 1)
        masked_scores = best_scores.masked_fill(~valid_best, torch.finfo(best_scores.dtype).min)
        weights = torch.softmax(masked_scores, dim=-1).to(device="cpu")
        all_values = self._grid[torch.arange(start, max_end, dtype=torch.long)]
        all_values = all_values.reshape(-1, nodes).transpose(0, 1)
        indices = (best_times - start).clamp_min(0).to(device="cpu")
        values = all_values.unsqueeze(0).expand(len(uncached_positions), -1, -1).gather(2, indices)
        retrieved = (weights * values).sum(dim=-1)
        retrieved[~has_candidates] = 0.0

        for row, position, target_time in zip(retrieved, uncached_positions, uncached_times):
            result[position] = row
            if self._cache is not None and 0 <= target_time < total_steps:
                self._cache[target_time] = row
        return result.to(device=local_crop.device)


__all__ = ["CausalRetrieval"]
