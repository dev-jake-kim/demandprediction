from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GridDemandDataset(Dataset):
    """전처리 파이프라인(`preprocessing/*/create_graph.py`)이 저장한
    `temporal_grid.npy`((T, X, Y), 시간대별 격자 수요)를 읽어,
    직전 `time_step`시간(t-k ~ t-1)의 전체 격자 수요로 특정 노드의 t시점 수요를
    예측하는 샘플을 만든다.

    반환 형식은 HuggingFace `Trainer`에서 바로 쓸 수 있도록
    `{'demands', 'labels', 'node_id', 'sample_idx'}` 딕셔너리로 고정한다
    (configs/config.yaml의 `remove_unused_columns: false`와 짝을 맞춤 —
    `node_id`/`sample_idx`는 모델 forward에 안 쓰여도 collate 단계에서 제거되지 않음).
    """

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
        self.n_nodes = self.X * self.Y

    def __len__(self) -> int:
        return self.n_t * self.n_nodes

    def __getitem__(self, idx: int) -> dict:
        t_pos, spatial_idx = divmod(idx, self.n_nodes)
        x_idx, y_idx = divmod(spatial_idx, self.Y)
        t = self.t_start + t_pos

        demand_seq = self.grid[t - self.time_step:t]  # (time_step, X, Y)
        label = self.grid[t, x_idx, y_idx]  # scalar

        return {
            'demands': torch.from_numpy(demand_seq),
            'labels': torch.tensor(label, dtype=torch.float32),
            'node_id': torch.tensor(x_idx * self.Y + y_idx, dtype=torch.long),
            'sample_idx': torch.tensor(idx, dtype=torch.long),
        }
