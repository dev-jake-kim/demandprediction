from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WEATHER_COLUMNS = ['기온(°C)', '강수량(mm)', '적설(cm)']
ZERO_FILL_WEATHER_COLUMNS = ['강수량(mm)', '적설(cm)']  # 관측 없음=0인 컬럼만 (기온은 zero-fill 대상 아님)
DAY_HOURS = 24
WEEK_HOURS = 168


class GridDemandDataset(Dataset):
    """전처리 파이프라인(`preprocessing/*/create_graph.py`)이 저장한
    `temporal_grid.npy`((T, X, Y), 시간대별 격자 수요)를 읽어,
    직전 `time_step`시간(t-k ~ t-1)의 전체 격자 수요로 t시점의 전체 격자(X*Y개 노드) 수요를
    한 번에 예측하는 샘플을 만든다. 샘플 하나 = 시간 t 하나 (노드별로 나누지 않음 —
    노드별 forward/backward는 `GridDemandModel`이 배치 차원을 늘려서 벡터화로 처리한다).

    반환 형식은 HuggingFace `Trainer`에서 바로 쓸 수 있도록
    `{'demands', 'labels', 'sample_idx', 'weather', 'hour_of_day', 'day_of_week'}` 딕셔너리로
    고정한다(configs/config.yaml의 `remove_unused_columns: false`와 짝을 맞춤).

    `sample_idx`는 이 Dataset 인스턴스 내부의 0-base idx가 아니라 **절대 시간 인덱스**
    (`t = t_start + idx`, 원본 npy 배열 기준)다. train/val/test가 전부 같은 근본 grid의
    서로 다른 `t_start`/`t_end` 슬라이스이므로, 절대 인덱스를 써야 어느 split에서 온 샘플이든
    같은 좌표계로 정렬된다(검색(retrieval) 브랜치가 있는 다른 모델 변형이 이 인과적 prefix
    계산에 사용 — `GridDemandModel`(assemble-no-ir)은 이 필드를 받기만 하고 쓰지 않는다).

    `weather`는 `weather_csv_path`의 기온/강수량/적설 3개 컬럼을 시간 인덱스로 그대로 정렬해
    읽은 것이다. **수요 윈도우([t-time_step, t-1])보다 한 칸 밀린 [t-time_step+1, t]를 쓴다** —
    예측 대상 시점 t 자체의 날씨까지 포함하는 의도적 설계로, "그 시점의 날씨 예보는 이미 안다"는
    가정이다(수요 자체를 미리 아는 것과는 다름 — 날씨는 단기 예보 정확도가 높아 이런 가정이
    현실적이라고 보고 채택함). `hour_of_day`/`day_of_week`는 캘린더 정보라 예보가 필요 없으므로
    수요와 동일한 비이동 윈도우 [t-time_step, t-1]을 쓴다(ADFormer 브랜치와 동일 계산 방식).

    `daily_demands`/`weekly_demands`는 "정확히 같은 시각"의 과거 `daily_lag_count`일치/
    `weekly_lag_count`주치 수요를 오래된 것부터 최근 것 순으로 담은 것이다(예:
    `daily_lag_count=6`이면 `[t-144, t-120, ..., t-24]`). recent 윈도우([t-time_step, t-1])와는
    독립적인 별도 시퀀스이며, 가장 먼 daily lag(`t - daily_lag_count*24`)와 weekly lag
    (`t - weekly_lag_count*168`) 둘 다 항상 0 이상이 되도록 `t_start`가 제한된다(아래 참고) —
    부족분을 0-패딩하지 않고 애초에 무효한 시점을 샘플로 만들지 않는, 이 클래스의 기존 관례
    (예: `weather_csv_path` 요구사항)를 그대로 따른다.
    """

    def __init__(
        self,
        npy_path: str | Path,
        time_step: int,
        weather_csv_path: str | Path,
        daily_lag_count: int = 6,
        weekly_lag_count: int = 4,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> None:
        if daily_lag_count <= 0 or weekly_lag_count <= 0:
            raise ValueError(
                f"daily_lag_count/weekly_lag_count는 1 이상이어야 함: "
                f"got daily_lag_count={daily_lag_count}, weekly_lag_count={weekly_lag_count}"
            )

        self.grid = np.load(npy_path).astype(np.float32)  # (T, X, Y)
        self.time_step = time_step
        self.daily_lag_count = daily_lag_count
        self.weekly_lag_count = weekly_lag_count
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

        # 가장 먼 daily lag(t - daily_lag_count*DAY_HOURS)와 weekly lag(t - weekly_lag_count*
        # WEEK_HOURS) 둘 다 항상 유효해야 하므로, time_step보다 이 둘 중 더 큰 쪽이 크면 그쪽으로
        # min_t_start를 올린다(0-패딩 없이 무효 구간 자체를 샘플에서 제외 — 클래스 docstring 참고).
        # daily_lag_count가 커지면(예: 30일치) weekly_lag_count*168보다 daily_lag_count*24가 더
        # 클 수 있으므로 둘 다 반영해야 함(둘 중 하나만 보면 음수 인덱스가 나와 numpy가 조용히
        # 배열 끝을 가리키는 wrap-around가 생겨 미래 데이터가 새어 들어갈 위험이 있음).
        min_t_start = max(self.time_step, self.daily_lag_count * DAY_HOURS, self.weekly_lag_count * WEEK_HOURS)
        self.t_start = min_t_start if t_start is None else max(t_start, min_t_start)
        self.t_end = self.T if t_end is None else min(t_end, self.T)
        if self.t_start >= self.t_end:
            raise ValueError(
                f"유효한 target 시간 구간이 없음: t_start={self.t_start}, t_end={self.t_end} "
                f"(time_step={self.time_step}, daily_lag_count={self.daily_lag_count}, "
                f"weekly_lag_count={self.weekly_lag_count}로 최소 {min_t_start} 이상 필요)"
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

        # daily/weekly 주기 lag: "정확히 같은 시각"의 과거 값, 오래된 것부터 최근 것 순.
        daily_idx = t - DAY_HOURS * np.arange(self.daily_lag_count, 0, -1)  # [t-144,...,t-24]
        weekly_idx = t - WEEK_HOURS * np.arange(self.weekly_lag_count, 0, -1)  # [t-672,...,t-168]
        daily_demands = self.grid[daily_idx]  # (daily_lag_count, X, Y)
        weekly_demands = self.grid[weekly_idx]  # (weekly_lag_count, X, Y)

        return {
            'demands': torch.from_numpy(demand_seq),
            'labels': torch.from_numpy(label),
            'sample_idx': torch.tensor(t, dtype=torch.long),
            'weather': torch.from_numpy(weather_seq),
            'hour_of_day': torch.from_numpy(hour_of_day),
            'day_of_week': torch.from_numpy(day_of_week),
            'daily_demands': torch.from_numpy(daily_demands),
            'weekly_demands': torch.from_numpy(weekly_demands),
        }
