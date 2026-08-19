from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import pandas as pd
from tqdm import tqdm


def remove_other_region(
    df: pd.DataFrame,
    x_col: str = 'xpos',
    y_col: str = 'ypos',
    x_range: tuple[int, int] = (128_950739, 129_500739),
    y_range: tuple[int, int] = (35_756099, 35_305729),
) -> pd.DataFrame:
    for column in (x_col, y_col):
        if column not in df.columns:
            raise KeyError(f"컬럼 없음: {column}")

    with tqdm(total=3, desc='remove_other_region', unit='step') as progress:
        mask = (
            (df[x_col] >= x_range[0]) & (df[x_col] <= x_range[1]) &
            (df[y_col] >= y_range[1]) & (df[y_col] <= y_range[0])
        )
        progress.update(1)
        filtered = df.loc[mask].copy()
        progress.update(1)
        tqdm.write(
            f"remove_other_region: kept {len(filtered)} / {len(df)} rows, "
            f"remain {len(filtered) / len(df) * 100:.2f}% data"
        )
        progress.update(1)

    return filtered


def eleminate_duplicates(
    df: pd.DataFrame,
    client_col: str = 'clientid',
    time_col: str = 'call_date',
    window_seconds: int = 600,
) -> pd.DataFrame:
    for column in (client_col, time_col):
        if column not in df.columns:
            raise KeyError(f"컬럼 없음: {column}")

    dropped_indices: list[int] = []
    queue: deque[tuple[object, Optional[pd.Timestamp]]] = deque()
    in_window_count: defaultdict[object, int] = defaultdict(int)

    def parse_timestamp(value) -> Optional[pd.Timestamp]:
        timestamp = pd.to_datetime(value, errors='coerce')
        if pd.isna(timestamp):
            return None
        return timestamp

    for index, row in tqdm(df.iterrows(), total=len(df), desc='eleminate_duplicates', unit='row'):
        client = row[client_col]
        timestamp = parse_timestamp(row[time_col])

        if timestamp is not None:
            while queue and (queue[0][1] is not None) and (timestamp - queue[0][1]).total_seconds() > window_seconds:
                old_client, _ = queue.popleft()
                in_window_count[old_client] -= 1
                if in_window_count[old_client] <= 0:
                    del in_window_count[old_client]

        if (timestamp is not None) and in_window_count.get(client, 0) > 0:
            dropped_indices.append(index)
            continue

        if timestamp is not None:
            queue.append((client, timestamp))
            in_window_count[client] += 1

    return df.drop(index=dropped_indices)


def crop_filter(df: pd.DataFrame, threshold_rate: float = 0.85, step_rate: float = 0.001) -> pd.DataFrame:
    for column in ('xpos', 'ypos'):
        if column not in df.columns:
            raise KeyError(f"컬럼 없음: {column}")

    points: list[list[int | float]] = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc='crop: build datas', unit='row'):
        points.append([row['xpos'], row['ypos'], 0, index])

    threshold = int(len(points) * threshold_rate)

    def refine_center(items: list[list[int | float]]) -> tuple[int, int]:
        x_sum = 0
        y_sum = 0
        for x_value, y_value, _, _ in items:
            x_sum += x_value
            y_sum += y_value
        return x_sum // len(items), y_sum // len(items)

    def delete_outliers(x_mean: int, y_mean: int, items: list[list[int | float]], rate: float) -> None:
        for item in items:
            delta_x = abs(item[0] - x_mean)
            delta_y = abs(item[1] - y_mean)
            item[2] = max(delta_x, delta_y)
        items.sort(key=lambda value: value[2])
        cut_length = int(len(items) * rate)
        if cut_length > 0:
            del items[-cut_length:]

    iteration = 0
    with tqdm(desc='crop: iter', unit='iter') as progress:
        while len(points) > threshold and len(points) > 0:
            iteration += 1
            x_mid, y_mid = refine_center(points)
            delete_outliers(x_mid, y_mid, points, rate=step_rate)
            progress.set_postfix({'remain': len(points), 'target': threshold})
            progress.update(1)
            if step_rate == 0 or iteration > 50000:
                break

    remaining_indices = [int(item[3]) for item in points]
    cropped = df.loc[remaining_indices].copy()
    tqdm.write(f"crop_filter: kept {len(cropped)} / {len(df)} rows")
    return cropped
