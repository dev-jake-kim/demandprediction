from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GridDemandDataset(Dataset):
    """전처리 파이프라인(`preprocessing/*/create_graph.py`)이 저장한
    `temporal_grid.npy`((T, X, Y), 시간대별 격자 수요)를 읽어,
    직전 `time_step`시간(t-k ~ t-1)의 전체 격자 수요로 t시점의 전체 격자(X*Y개 노드) 수요를
    한 번에 예측하는 샘플을 만든다. 샘플 하나 = 시간 t 하나.

    `hour_of_day`/`day_of_week`는 npy의 절대 시간 인덱스 t로부터 산술적으로만 계산한다
    (`t % 24`, `(t // 24) % 7`) — 실제 달력 날짜에 맞출 필요 없음. DMVST-Net은 공휴일을 쓰지
    않고 요일의 "주기성"만 학습하므로, t=0을 무슨 요일로 보든 같은 실제 요일이 항상 같은
    label로 일관되게 매핑되기만 하면 학습에 영향이 없다(ADFormer/STResnet과 동일 근거).

    반환 형식은 HuggingFace `Trainer`에서 바로 쓸 수 있도록
    `{'demands', 'labels', 'hour_of_day', 'day_of_week', 'sample_idx'}` 딕셔너리로 고정한다
    (configs/config.yaml의 `remove_unused_columns: false`와 짝을 맞춤).
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

    def __len__(self) -> int:
        return self.n_t

    def __getitem__(self, idx: int) -> dict:
        t = self.t_start + idx

        demand_seq = self.grid[t - self.time_step:t]  # (time_step, X, Y)
        label = self.grid[t]  # (X, Y) - 전체 노드

        abs_hours = np.arange(t - self.time_step, t)  # lookback 구간의 절대 시간 인덱스
        hour_of_day = (abs_hours % 24).astype(np.int64)
        day_of_week = ((abs_hours // 24) % 7).astype(np.int64)

        return {
            'demands': torch.from_numpy(demand_seq),
            'labels': torch.from_numpy(label),
            'hour_of_day': torch.from_numpy(hour_of_day),
            'day_of_week': torch.from_numpy(day_of_week),
            'sample_idx': torch.tensor(idx, dtype=torch.long),
        }
