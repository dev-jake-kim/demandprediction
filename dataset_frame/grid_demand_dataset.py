from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PERIOD = 24  # 하루(시간) — 논문의 p
TREND_SPAN = 24 * 7  # 일주일(시간) — 논문의 q


class GridDemandDataset(Dataset):
    """전처리 파이프라인(`preprocessing/*/create_graph.py`)이 저장한
    `temporal_grid.npy`((T, X, Y), 시간대별 격자 수요)를 읽어, ST-ResNet이 요구하는
    closeness/period/trend 세 시퀀스 + 예측 시점 t의 외부 요인(day_of_week)으로 샘플을 만든다.

    - closeness: t 직전 l_c개 연속 시간
    - period: 하루(PERIOD) 간격으로 l_p개 (t-PERIOD, t-2*PERIOD, ...)
    - trend: 일주일(TREND_SPAN) 간격으로 l_q개 (t-TREND_SPAN, t-2*TREND_SPAN, ...)
    - day_of_week: t로부터 산술 계산(`(t//24)%7`) — 실제 달력에 맞출 필요 없음(공휴일 미사용,
      요일 주기성만 학습에 쓰이므로 t=0을 무슨 요일로 보든 전역적으로 일관되면 무해함).

    반환 형식은 HuggingFace `Trainer`에서 바로 쓸 수 있도록
    `{'demands_closeness', 'demands_period', 'demands_trend', 'day_of_week', 'labels', 'sample_idx'}`
    딕셔너리로 고정한다(configs/config.yaml의 `remove_unused_columns: false`와 짝을 맞춤).
    """

    def __init__(
        self,
        npy_path: str | Path,
        l_c: int,
        l_p: int,
        l_q: int,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> None:
        self.grid = np.load(npy_path).astype(np.float32)  # (T, X, Y)
        self.l_c, self.l_p, self.l_q = l_c, l_p, l_q
        self.T, self.X, self.Y = self.grid.shape

        min_t_start = max(l_c, l_p * PERIOD, l_q * TREND_SPAN)
        self.t_start = min_t_start if t_start is None else max(t_start, min_t_start)
        self.t_end = self.T if t_end is None else min(t_end, self.T)
        if self.t_start >= self.t_end:
            raise ValueError(
                f"유효한 target 시간 구간이 없음: t_start={self.t_start}, t_end={self.t_end} "
                f"(l_c={l_c}, l_p={l_p}*{PERIOD}, l_q={l_q}*{TREND_SPAN}로 최소 {min_t_start} 이상 필요)"
            )

        self.n_t = self.t_end - self.t_start

    def __len__(self) -> int:
        return self.n_t

    def __getitem__(self, idx: int) -> dict:
        t = self.t_start + idx

        label = self.grid[t]  # (X, Y)
        day_of_week = (t // 24) % 7

        sample = {
            'day_of_week': torch.tensor(day_of_week, dtype=torch.long),
            'labels': torch.from_numpy(label),
            'sample_idx': torch.tensor(idx, dtype=torch.long),
        }

        # l_c/l_p/l_q=0이면 해당 시퀀스는 키 자체를 안 넣는다 — 모델이 그 브랜치를 비활성화했을 때
        # forward()의 Optional 인자가 기본값(None)으로 채워지도록.
        if self.l_c > 0:
            sample['demands_closeness'] = torch.from_numpy(self.grid[t - self.l_c:t])  # (l_c,X,Y), 오래된 순
        if self.l_p > 0:
            period_idx = [t - PERIOD * i for i in range(self.l_p, 0, -1)]
            sample['demands_period'] = torch.from_numpy(self.grid[period_idx])  # (l_p,X,Y), 오래된 순
        if self.l_q > 0:
            trend_idx = [t - TREND_SPAN * i for i in range(self.l_q, 0, -1)]
            sample['demands_trend'] = torch.from_numpy(self.grid[trend_idx])  # (l_q,X,Y), 오래된 순

        return sample
