from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GridDemandDataset(Dataset):
    """Load a temporal demand grid and return history/label samples per target time."""

    def __init__(
        self,
        npy_path: str | Path,
        time_step: int,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> None:
        self.grid = np.load(npy_path).astype(np.float32)  # (T, X, Y)
        self.time_step = time_step
        self.T, self.X, self.Y = self.grid.shape

        min_t_start = self.time_step
        self.t_start = min_t_start if t_start is None else max(t_start, min_t_start)
        self.t_end = self.T if t_end is None else min(t_end, self.T)
        if self.t_start >= self.t_end:
            raise ValueError(
                f"유효한 target 시간 구간이 없음: t_start={self.t_start}, t_end={self.t_end} "
                f"(time_step={self.time_step}로 최소 {min_t_start} 이상 필요)"
            )

        self.n_t = self.t_end - self.t_start

    def __len__(self) -> int:
        return self.n_t

    def __getitem__(self, idx: int) -> dict:
        t = self.t_start + idx

        demand_seq = self.grid[t - self.time_step:t]  # (time_step, X, Y)
        label = self.grid[t]  # (X, Y) - 전체 노드

        return {
            'demands': torch.from_numpy(demand_seq),
            'labels': torch.from_numpy(label),
            'sample_idx': torch.tensor(idx, dtype=torch.long),
        }
