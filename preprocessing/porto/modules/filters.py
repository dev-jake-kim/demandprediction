from __future__ import annotations

from tqdm import tqdm

import geopandas as gpd
import pandas as pd


def remove_other_region(
    df: pd.DataFrame,
    x_col: str = 'xpos',
    y_col: str = 'ypos',
    x_range: tuple[int, int] = (-8_750000, -8_500000),
    y_range: tuple[int, int] = (41_250000, 41_050000),
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


def crop_to_projected_bounds(
    gdf: gpd.GeoDataFrame,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> gpd.GeoDataFrame:
    """투영된(target_crs, 미터 단위) 좌표 기준 사각 하드캡 필터.

    crop_filter(이상치 반복 제거)와 달리 반복적으로 다듬지 않고, 지정한
    x/y 범위(둘 다 (min, max) 순서) 밖의 점을 한 번에 잘라낸다.
    """
    x = gdf.geometry.x
    y = gdf.geometry.y
    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1])
    )
    filtered = gdf.loc[mask].copy()
    print(
        f"crop_to_projected_bounds: kept {len(filtered)} / {len(gdf)} rows, "
        f"remain {len(filtered) / len(gdf) * 100:.2f}% data"
    )
    return filtered


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
