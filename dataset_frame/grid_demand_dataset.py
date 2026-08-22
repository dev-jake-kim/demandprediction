from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GridDemandDataset(Dataset):
    """전처리 파이프라인(`preprocessing/*/create_graph.py`)이 저장한
    `temporal_grid.npy`((T, X, Y), 시간대별 격자 수요)를 읽어,
    직전 `time_step`시간(t-k ~ t-1)의 전체 격자 수요로 t시점의 전체 격자(X*Y개 노드) 수요를
    한 번에 예측하는 샘플을 만든다. 샘플 하나 = 시간 t 하나 (노드별로 나누지 않음 —
    노드별 forward/backward는 `GridDemandModel`이 배치 차원을 늘려서 벡터화로 처리한다).

    반환 형식은 HuggingFace `Trainer`에서 바로 쓸 수 있도록
    `{'demands', 'labels', 'sample_idx'}` 딕셔너리로 고정한다
    (configs/config.yaml의 `remove_unused_columns: false`와 짝을 맞춤).

    `sample_idx`는 이 Dataset 인스턴스 내부의 0-base idx가 아니라 **절대 시간 인덱스**
    (`t = t_start + idx`, 원본 npy 배열 기준)다. train/val/test가 전부 같은 근본 grid의
    서로 다른 `t_start`/`t_end` 슬라이스이므로, 절대 인덱스를 써야 어느 split에서 온
    샘플이든 검색(retrieval) 브랜치의 인과적 prefix 계산에 그대로 정렬돼서 쓸 수 있다.
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

        return {
            'demands': torch.from_numpy(demand_seq),
            'labels': torch.from_numpy(label),
            'sample_idx': torch.tensor(t, dtype=torch.long),
        }
