"""Temporal-grid dataset and causal daily/weekly lag construction (merged model).

The dataset deliberately works with raw demand values.  The model owns the
log1p transformations for the neural, daily, and weekly branches so that the
retrieval branch can remain on the raw scale.

원래 `merged_model/data.py`에 있던 것을 이 저장소의 `dataset_frame/` 관례로 옮긴 것이다.
계산은 그대로이고, 두 가지만 바뀌었다:

* 반환 dict의 ``"target"`` -> ``"labels"`` (``GridDemandDataset``/HF ``Trainer`` 관례).
* ``resolve_dataset_path``의 탐색 경로가 이 저장소 레이아웃(``<repo>/data/raw``) 기준이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATASET_FILES = {
    'ulsan': 'ulsan_temporal_grid.npy',
    'porto': 'porto_temporal_grid.npy',
}

# 날씨 CSV 규약은 dataset_frame/grid_demand_dataset.py와 동일하다.
WEATHER_COLUMNS = ['기온(°C)', '강수량(mm)', '적설(cm)']
# 기상청 관측 데이터 관례상 강수량/적설의 빈 셀은 "관측값 없음"이 아니라 0을 뜻한다
# (강수/적설이 없으면 값을 아예 안 채움). 기온은 이 관례가 적용되지 않으므로 제외한다.
ZERO_FILL_WEATHER_COLUMNS = ['강수량(mm)', '적설(cm)']
NUM_WEATHER_FEATURES = len(WEATHER_COLUMNS)

# dataset_frame/ 의 부모 = 저장소 루트.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _unique_existing(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path not in result and path.is_file():
            result.append(path)
    return result


def resolve_dataset_path(dataset: str, explicit_path: str | Path | None = None) -> Path:
    """Resolve a temporal-grid file without falling back to legacy demand.npy.

    명시 경로가 있으면 cwd 기준 -> 저장소 루트 기준 순으로 보고, 없으면 이 저장소의
    표준 위치(``<repo>/data/raw/<city>_temporal_grid.npy``)를 쓴다.
    """

    dataset = dataset.lower()
    if dataset not in DATASET_FILES:
        raise ValueError(f'Unsupported dataset {dataset!r}; choose from {sorted(DATASET_FILES)}')
    filename = DATASET_FILES[dataset]

    candidates: list[Path] = []
    if explicit_path is not None:
        requested = Path(explicit_path).expanduser()
        candidates.append(requested if requested.is_absolute() else (Path.cwd() / requested))
        candidates.append(requested if requested.is_absolute() else (REPO_ROOT / requested))
    candidates.append(REPO_ROOT / 'data' / 'raw' / filename)
    existing = _unique_existing(candidates)
    if existing:
        return existing[0]

    searched = '\n  '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'No temporal-grid file found for {dataset!r}. Searched:\n  {searched}\n'
        'The merged model intentionally does not fall back to legacy demand.npy.'
    )


def _chronological_lags(period: int, count: int, radius: int) -> np.ndarray:
    """Return source lags from oldest to newest in chronological order."""

    if period <= 0 or count <= 0 or radius < 0:
        raise ValueError('period and count must be positive; radius must be non-negative')
    lags = {
        base + offset
        for base in (period * i for i in range(1, count + 1))
        for offset in range(-radius, radius + 1)
        if base + offset > 0
    }
    return np.asarray(sorted(lags, reverse=True), dtype=np.int64)


def load_weather_table(weather_csv_path: str | Path, total_steps: int) -> np.ndarray:
    """날씨 CSV를 ``[T, 3]``(기온/강수량/적설)로 읽는다.

    cp949 인코딩, 강수량/적설만 결측을 0으로 채운다. 행 수가 temporal grid의 길이와 다르면
    시간 정렬이 깨진 것이므로 즉시 실패시킨다.
    """

    path = Path(weather_csv_path).expanduser().resolve()
    frame = pd.read_csv(path, encoding='cp949')
    missing = [column for column in WEATHER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f'날씨 CSV에 필요한 컬럼이 없음: {missing} ({path}). '
            f'있는 컬럼: {list(frame.columns)}'
        )
    frame = frame.copy()
    frame[ZERO_FILL_WEATHER_COLUMNS] = frame[ZERO_FILL_WEATHER_COLUMNS].fillna(0.0)
    weather = frame[WEATHER_COLUMNS].to_numpy(dtype=np.float32)
    if weather.shape[0] != total_steps:
        raise ValueError(
            f'날씨 행 수({weather.shape[0]})가 temporal grid 길이({total_steps})와 다름: {path}'
        )
    if not np.isfinite(weather).all():
        raise ValueError(f'날씨에 결측/비유한값이 남아 있음: {path}')
    return weather


def _calendar_features(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """절대 시간 인덱스 -> (hour_of_day, day_of_week).

    인덱스 기반 합성 캘린더다(실제 달력 날짜가 아님) — 저장소의 다른 브랜치들과 같은 값을
    쓰기 위해 공식을 그대로 맞춘다.
    """

    hour = (times % 24).astype(np.int64)
    day_of_week = ((times // 24) % 7).astype(np.int64)
    return hour, day_of_week


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    total: int


class UnifiedDemandDataset(Dataset):
    """One sample per target time from a ``[T, H, W]`` temporal grid.

    ``daily_mask`` and ``weekly_mask`` use the convention required by the
    model: ``True`` means that the lag is invalid.  ``sample_idx`` is always
    the absolute index in the original temporal grid, never a split-local
    index.

    반환 dict의 키는 ``MergedDemandModel.forward``의 인자명과 1:1로 맞춰져 있다
    (``configs/config.yaml``의 ``remove_unused_columns: false``와 짝).
    """

    def __init__(
        self,
        data_path: str | Path,
        split: Literal['train', 'val', 'test'] = 'train',
        *,
        weather_csv_path: str | Path,
        time_step: int = 24,
        daily_period: int = 24,
        daily_lags: int = 6,
        weekly_period: int = 24 * 7,
        weekly_lags: int = 4,
        lag_radius: int = 0,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
            raise ValueError('train_ratio and val_ratio must leave a non-empty test split')
        if time_step <= 0:
            raise ValueError('time_step must be positive')

        self.data_path = Path(data_path).expanduser().resolve()
        grid = np.load(self.data_path, mmap_mode='r')
        if grid.ndim != 3:
            raise ValueError(f'Expected temporal grid [T,H,W], got {grid.shape} from {self.data_path}')
        if not np.issubdtype(grid.dtype, np.number):
            raise TypeError(f'Temporal grid must be numeric, got {grid.dtype}')
        if not np.isfinite(grid).all():
            raise ValueError(f'Temporal grid contains non-finite values: {self.data_path}')
        if np.min(grid) < 0:
            raise ValueError('Demand must be non-negative for log1p and Softplus output')

        self.grid = np.asarray(grid, dtype=np.float32)
        self.total_steps, self.height, self.width = self.grid.shape
        self.num_nodes = self.height * self.width
        self.time_step = int(time_step)
        self.daily_lag_values = _chronological_lags(daily_period, daily_lags, lag_radius)
        self.weekly_lag_values = _chronological_lags(weekly_period, weekly_lags, lag_radius)

        train_end = int(self.total_steps * train_ratio)
        val_end = int(self.total_steps * (train_ratio + val_ratio))
        self.bounds = SplitBounds(train_end=train_end, val_end=val_end, total=self.total_steps)
        split_starts = {
            'train': self.time_step,
            'val': max(self.time_step, train_end),
            'test': max(self.time_step, val_end),
        }
        split_ends = {'train': train_end, 'val': val_end, 'test': self.total_steps}
        if split not in split_starts:
            raise ValueError(f'Unsupported split {split!r}')
        start, end = split_starts[split], split_ends[split]
        if end <= start:
            raise ValueError(f'Split {split!r} is empty: [{start}, {end})')
        self.split = split
        self.indices = np.arange(start, end, dtype=np.int64)

        self.weather = load_weather_table(weather_csv_path, self.total_steps)
        self.weather_csv_path = Path(weather_csv_path).expanduser().resolve()
        all_times = np.arange(self.total_steps, dtype=np.int64)
        self.hour_table, self.day_of_week_table = _calendar_features(all_times)

        self.daily_values, self.daily_mask = self._make_lag_table(self.daily_lag_values)
        self.weekly_values, self.weekly_mask = self._make_lag_table(self.weekly_lag_values)
        self.daily_context = self._make_lag_context(self.daily_lag_values)
        self.weekly_context = self._make_lag_context(self.weekly_lag_values)

    def _make_lag_context(self, lags: np.ndarray) -> dict[str, np.ndarray]:
        """lag 시점별 날씨/캘린더를 ``[T, L, ...]``로 미리 만든다.

        무효 lag(``source_time < 0``)는 ``_make_lag_table``이 수요를 0으로 채우는 것과 같은
        방식으로 0을 채운다. 어차피 PeriodicLSTMEncoder의 compaction에서 제외되지만,
        ``safe_times`` clip 때문에 그냥 두면 엉뚱한 시점의 값이 들어가므로 방어적으로 지운다.
        """

        source_times = np.arange(self.total_steps, dtype=np.int64)[:, None] - lags[None, :]
        valid = source_times >= 0
        safe_times = np.clip(source_times, 0, self.total_steps - 1)

        weather = np.where(valid[..., None], self.weather[safe_times], 0.0).astype(np.float32)
        hour = np.where(valid, self.hour_table[safe_times], 0).astype(np.int64)
        day_of_week = np.where(valid, self.day_of_week_table[safe_times], 0).astype(np.int64)
        return {
            'weather': np.ascontiguousarray(weather),
            'hour': np.ascontiguousarray(hour),
            'day_of_week': np.ascontiguousarray(day_of_week),
        }

    def _make_lag_table(self, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_times = np.arange(self.total_steps, dtype=np.int64)[:, None] - lags[None, :]
        valid = source_times >= 0
        safe_times = np.clip(source_times, 0, self.total_steps - 1)
        # [T, L, H, W] -> [T, L, N, 1].
        values = self.grid[safe_times].reshape(self.total_steps, len(lags), self.num_nodes)
        values = np.where(valid[..., None], values, 0.0).astype(np.float32, copy=False)
        values = values[..., None]
        return np.ascontiguousarray(values), np.ascontiguousarray(~valid)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_time = int(self.indices[item])
        history = self.grid[target_time - self.time_step : target_time]
        target = self.grid[target_time]

        # 날씨는 수요 윈도우보다 한 칸 밀린 [t-k+1, t+1)을 쓴다 — 예측 시점 t의 날씨까지 포함하는
        # 의도적 설계로, "단기 날씨 예보는 이미 안다"는 가정을 그대로 따른다(수요를 미리 아는
        # 것과는 다름). 캘린더는 예보가 필요 없으므로 수요와 같은 [t-k, t)를 쓴다.
        recent_weather = self.weather[target_time - self.time_step + 1 : target_time + 1]
        recent_times = np.arange(target_time - self.time_step, target_time, dtype=np.int64)
        recent_hour, recent_day_of_week = _calendar_features(recent_times)

        return {
            'demand_history': torch.from_numpy(np.array(history, dtype=np.float32, copy=True)),
            'daily_demand': torch.from_numpy(self.daily_values[target_time]),
            'daily_mask': torch.from_numpy(self.daily_mask[target_time]),
            'weekly_demand': torch.from_numpy(self.weekly_values[target_time]),
            'weekly_mask': torch.from_numpy(self.weekly_mask[target_time]),
            # 원본 merged_model의 "target". HF Trainer의 label_names 기본값과 맞추려고
            # 이름만 "labels"로 바꿨다(값/계산은 동일).
            'labels': torch.from_numpy(np.array(target, dtype=np.float32, copy=True)),
            'sample_idx': torch.tensor(target_time, dtype=torch.long),
            'weather': torch.from_numpy(np.array(recent_weather, dtype=np.float32, copy=True)),
            'hour_of_day': torch.from_numpy(recent_hour),
            'day_of_week': torch.from_numpy(recent_day_of_week),
            'daily_weather': torch.from_numpy(self.daily_context['weather'][target_time]),
            'daily_hour': torch.from_numpy(self.daily_context['hour'][target_time]),
            'daily_day_of_week': torch.from_numpy(self.daily_context['day_of_week'][target_time]),
            'weekly_weather': torch.from_numpy(self.weekly_context['weather'][target_time]),
            'weekly_hour': torch.from_numpy(self.weekly_context['hour'][target_time]),
            'weekly_day_of_week': torch.from_numpy(self.weekly_context['day_of_week'][target_time]),
        }

    @property
    def train_end(self) -> int:
        return self.bounds.train_end

    @property
    def val_end(self) -> int:
        return self.bounds.val_end


__all__ = [
    'NUM_WEATHER_FEATURES',
    'WEATHER_COLUMNS',
    'UnifiedDemandDataset',
    'load_weather_table',
    'resolve_dataset_path',
]
