from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WEATHER_COLUMNS = ['기온(°C)', '강수량(mm)', '적설(cm)']
ZERO_FILL_WEATHER_COLUMNS = ['강수량(mm)', '적설(cm)']  # 관측 없음=0인 컬럼만 (기온은 zero-fill 대상 아님)


class GridDemandDataset(Dataset):
    """전처리 파이프라인(`preprocessing/*/create_graph.py`)이 저장한
    `temporal_grid.npy`((T, X, Y), 시간대별 격자 수요)를 읽어,
    직전 `time_step`시간(t-k ~ t-1)의 전체 격자 수요로 t시점의 전체 격자(X*Y개 노드) 수요를
    한 번에 예측하는 샘플을 만든다. 샘플 하나 = 시간 t 하나 (노드별로 나누지 않음 —
    노드별 forward/backward는 `GridDemandModel`이 배치 차원을 늘려서 벡터화로 처리한다).

    반환 형식은 HuggingFace `Trainer`에서 바로 쓸 수 있도록
    `{'demands', 'labels', 'sample_idx', 'weather', 'hour_of_day', 'day_of_week'}` 딕셔너리로
    고정한다(configs/config.yaml의 `remove_unused_columns: false`와 짝을 맞춤).

    `weather`는 `weather_csv_path`의 기온/강수량/적설 3개 컬럼을 시간 인덱스로 그대로 정렬해
    읽은 것이다. **수요 윈도우([t-time_step, t-1])보다 한 칸 밀린 [t-time_step+1, t]를 쓴다** —
    예측 대상 시점 t 자체의 날씨까지 포함하는 의도적 설계로, "그 시점의 날씨 예보는 이미 안다"는
    가정이다(수요 자체를 미리 아는 것과는 다름 — 날씨는 단기 예보 정확도가 높아 이런 가정이
    현실적이라고 보고 채택함). `hour_of_day`/`day_of_week`는 캘린더 정보라 예보가 필요 없으므로
    수요와 동일한 비이동 윈도우 [t-time_step, t-1]을 쓴다(ADFormer 브랜치와 동일 계산 방식).
    """

    def __init__(
        self,
        npy_path: str | Path,
        time_step: int,
        weather_csv_path: str | Path,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> None:
        self.grid = np.load(npy_path).astype(np.float32)  # (T, X, Y)
        self.time_step = time_step
        self.T, self.X, self.Y = self.grid.shape

        weather_df = pd.read_csv(weather_csv_path, encoding='cp949')
        # 기상청 관측 데이터 관례상 강수량/적설 빈 셀은 "관측값 없음"이 아니라 "0"을 의미함
        # (강수/적설이 없으면 값을 아예 안 채움) -> pandas가 NaN으로 읽은 걸 0으로 채워야 함.
        # 기온은 이 관례가 적용되지 않으므로 zero-fill 대상에서 제외 (진짜 결측이면 아래에서 에러).
        weather_df = weather_df.copy()
        weather_df[ZERO_FILL_WEATHER_COLUMNS] = weather_df[ZERO_FILL_WEATHER_COLUMNS].fillna(0.0)
        self.weather = weather_df[WEATHER_COLUMNS].to_numpy(dtype=np.float32)  # (T, 3)
        if self.weather.shape[0] != self.T:
            raise ValueError(
                f"weather_csv_path의 행 수({self.weather.shape[0]})가 grid 길이(T={self.T})와 다름 — "
                f"{weather_csv_path}가 이 grid와 같은 기간을 가리키는지 확인해야 함"
            )
        if not np.isfinite(self.weather).all():
            bad_cols = [
                WEATHER_COLUMNS[j]
                for j in range(len(WEATHER_COLUMNS))
                if not np.isfinite(self.weather[:, j]).all()
            ]
            raise ValueError(
                f"weather_csv_path에 결측/무한값이 있음(컬럼: {bad_cols}) — "
                f"{weather_csv_path}의 데이터를 확인해야 함"
            )

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

        # 날씨: 수요 윈도우보다 한 칸 밀림(t 자체의 날씨까지 포함) — 클래스 docstring 참고.
        weather_seq = self.weather[t - self.time_step + 1:t + 1]  # (time_step, 3)

        # 캘린더: 수요와 동일한 비이동 윈도우. ADFormer 브랜치와 동일 계산.
        abs_hours = np.arange(t - self.time_step, t)
        hour_of_day = (abs_hours % 24).astype(np.int64)
        day_of_week = ((abs_hours // 24) % 7).astype(np.int64)

        return {
            'demands': torch.from_numpy(demand_seq),
            'labels': torch.from_numpy(label),
            'sample_idx': torch.tensor(idx, dtype=torch.long),
            'weather': torch.from_numpy(weather_seq),
            'hour_of_day': torch.from_numpy(hour_of_day),
            'day_of_week': torch.from_numpy(day_of_week),
        }
