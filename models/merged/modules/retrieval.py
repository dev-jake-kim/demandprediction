"""Raw local-window retrieval with a causal time boundary.

질의(``query_space``):

- ``local``: 노드마다 자기 주변 창 ``[t−k, t)``로 따로 검색한다(노드마다 고른 시점이 다르다).
  유사도는 기본이 raw 창 cosine이고, ``retrieval_encoder_path``가 있으면 단독 학습한 검색 encoder의
  latent 거리 ``−‖μ_q − μ_τ‖² / (L·T)``로 바꾼다(docs/RETRIEVAL_ENCODER.md).
- ``global``: 지도 전체 ``[t−k, t)``(``k·N`` 값)를 한 벡터로 펴 cosine top-k 시점을 고르고, 그 시점들을
  모든 노드가 공유한다. 노드마다 자기 위치의 ``grid[τ, n]``을 같은 가중치로 평균한다.

후보 범위·top-k·softmax 가중평균·값(노드 자신의 τ 시점 수요)은 두 질의에서 같다.
"""

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
        retrieval_encoder_path: str | Path | None = None,
        query_space: Literal["local", "global"] = "local",
    ) -> None:
        super().__init__()
        if retrieval_scope not in {"observed_past", "train_prefix"}:
            raise ValueError("retrieval_scope must be 'observed_past' or 'train_prefix'")
        if query_space not in {"local", "global"}:
            raise ValueError("retrieval_query must be 'local' or 'global'")
        if query_space == "global" and retrieval_encoder_path is not None:
            raise ValueError("retrieval_encoder_path는 노드별(local) 질의 전용이다")
        self.query_space = query_space
        # global 질의: 모든 t의 결과 [T, N]과 고른 시점·유사도 [T, K] (첫 forward에서 계산).
        self._global: tuple[Tensor, Tensor, Tensor] | None = None
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
        # encoder는 첫 forward에서 입력 장치에 읽는다(from_pretrained의 meta-device 초기화와 무관).
        self.encoder_path = None if retrieval_encoder_path is None else str(retrieval_encoder_path)
        self._encoded: Tensor | None = None  # [T, N, L] 모든 시점·노드의 μ
        self._encoder_scale = 1.0  # 1 / (L·T)
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
    def _encoded_table(self, device: torch.device) -> Tensor:
        """검색 encoder(eval, 잡음 없음)로 모든 ``t ≥ k``, 모든 노드의 μ ``[T, N, L]``를 한 번 계산한다."""

        if self._encoded is None or self._encoded.device != device:
            from models.retrieval_encoder import encode_all, load_retrieval_encoder

            assert self._crops is not None and self.encoder_path is not None
            encoder = load_retrieval_encoder(self.encoder_path, device)
            if encoder.time_step != self.time_step or encoder.num_neighbors != self.num_neighbors:
                raise ValueError(
                    f'검색 encoder 입력 (time_step={encoder.time_step}, 창 {encoder.num_neighbors}칸)이 '
                    f'검색기 (time_step={self.time_step}, 창 {self.num_neighbors}칸)와 다르다'
                )
            if encoder.node_embedding.shape[0] != self.num_nodes:
                raise ValueError('검색 encoder의 노드 수가 격자와 다르다')
            self._encoded = encode_all(encoder, self._crops.to(device), self.time_step)
            self._encoder_scale = 1.0 / (encoder.latent_dim * encoder.temperature)
        return self._encoded

    @torch.no_grad()
    def global_table(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """지도 전체 질의의 ``(retrieved [T, N], times [T, K], scores [T, K])``.

        질의·후보 벡터는 창 ``[τ−k, τ)``의 전체 격자를 편 ``[k·N]``을 L2 정규화한 것이다. 후보는
        ``τ ∈ [k, end)`` (``end = t`` 또는 ``min(t, retrieval_train_end)``)이고, 고른 시점 τ는 모든 노드가
        공유한다. 후보가 없는 행(``t ≤ k``)은 retrieved 0, times −1, scores −inf다.
        """

        device = torch.device(device)
        if self._global is not None and self._global[0].device == device:
            return self._global
        assert self._grid is not None
        grid = self._grid.to(device).reshape(self._grid.shape[0], -1)  # [T, N]
        total, k = grid.shape[0], self.time_step
        top = self.retrieval_k
        retrieved = grid.new_zeros(total, grid.shape[1])
        times = torch.full((total, top), -1, dtype=torch.long, device=device)
        scores = grid.new_full((total, top), float("-inf"))
        if total > k:
            windows = grid.unfold(0, k, 1)[: total - k]  # [C, N, k], 창 [c, c+k) → 시점 c + k
            keys = windows.reshape(total - k, -1)
            keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            limit = None
            if self.retrieval_scope == "train_prefix":
                if self.retrieval_train_end is None:
                    raise ValueError("retrieval_train_end is required for train_prefix retrieval")
                limit = int(self.retrieval_train_end)
            candidate_times = torch.arange(k, total, device=device)
            for start in range(0, total - k, self.retrieval_chunk_size):
                stop = min(start + self.retrieval_chunk_size, total - k)
                query_times = candidate_times[start:stop]
                similarity = keys[start:stop] @ keys.T  # [Q, C]
                end = query_times if limit is None else query_times.clamp_max(limit)
                allowed = candidate_times.unsqueeze(0) < end.unsqueeze(1)
                similarity = similarity.masked_fill(~allowed, float("-inf"))
                take = min(top, similarity.shape[-1])
                best_scores, best_index = similarity.topk(take, dim=-1)
                valid = torch.isfinite(best_scores)
                weights = torch.softmax(best_scores, dim=-1).nan_to_num(0.0)  # 후보 없음 → 0
                best_times = torch.where(valid, candidate_times[best_index], -1)
                values = grid[best_times.clamp_min(0)] * valid.unsqueeze(-1)  # [Q, K, N]
                retrieved[query_times] = (weights.unsqueeze(-1) * values).sum(dim=1)
                times[query_times, :take] = best_times
                scores[query_times, :take] = best_scores
        self._global = (retrieved, times, scores)
        return self._global

    @torch.no_grad()
    def forward(self, demands: Tensor, sample_idx: Tensor) -> Tensor:
        """``[B, k, H, W]`` 최근 수요로 인과 후보를 검색해 ``[B, N]`` raw 값을 반환한다."""

        if demands.ndim != 4 or tuple(demands.shape[1:]) != (self.time_step, self.height, self.width):
            raise ValueError("Unexpected retrieval query shape")
        if self.query_space == "global":
            if self._grid is None:
                return demands.new_zeros((demands.shape[0], self.num_nodes))
            retrieved, _, _ = self.global_table(demands.device)
            index = sample_idx.to(device=demands.device, dtype=torch.long).clamp(0, retrieved.shape[0] - 1)
            return retrieved[index]
        local_crop = self._crop(demands)
        batch, steps, nodes, neighbors = local_crop.shape
        if self._grid is None or self._crops is None:
            return local_crop.new_zeros((batch, nodes))

        search_device = local_crop.device
        sample_times = sample_idx.detach().to(device="cpu", dtype=torch.long).tolist()
        encoded: Tensor | None = None
        if self.encoder_path is not None:
            encoded = self._encoded_table(search_device)
            # μ[t]는 창 [t−k, t)의 embedding = 이 샘플의 질의(demand_history와 같은 창).
            query_times = torch.as_tensor(sample_times, device=search_device).clamp(0, encoded.shape[0] - 1)
            query = encoded[query_times]
        else:
            query = local_crop.detach().to(device=search_device, dtype=torch.float32)
            query = query.permute(0, 2, 3, 1).reshape(batch, nodes, self.flat_query_dim)
            query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-8)
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
            if encoded is not None:
                # s = −‖μ_q − μ_τ‖² / (L·T), μ_τ = 창 [τ−k, τ)의 embedding.
                candidates = encoded[chunk_start:chunk_end].transpose(0, 1)  # [N, C, L]
                cross = torch.einsum("bnd,ncd->bnc", query_batch, candidates)
                distance_sq = (
                    query_batch.pow(2).sum(-1, keepdim=True)
                    + candidates.pow(2).sum(-1).unsqueeze(0)
                    - 2.0 * cross
                ).clamp_min(0.0)
                similarity = -distance_sq * self._encoder_scale
            else:
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
